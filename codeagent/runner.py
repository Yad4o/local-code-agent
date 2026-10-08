"""The agent loop. The HARNESS drives; the model only answers narrow questions.

For every plan step (test-driven, like a careful human pair):
  1. ask the model for the TEST files; they must fail before any code exists
  2. ask for the IMPLEMENTATION files; run the tests; on failure feed the real
     output back and retry (bounded)
  3. commit exactly the step's files, record progress
Anything that goes wrong rolls the step's files back to how they were.
"""
import ast
import shlex
import time
from dataclasses import dataclass, field
from typing import Callable

from codeagent import git
from codeagent.config import Config
from codeagent.llm import LLMClient, LLMError
from codeagent.parsing import parse_file_blocks
from codeagent.plan import Plan, PlanError, Step, plan_slug
from codeagent.prompts import build_impl_prompt, build_tests_prompt
from codeagent.replylog import ReplyLog
from codeagent.resources import ResourceGuard
from codeagent.shell import CommandResult, run_command
from codeagent.state import RunState, WorkspaceLock
from codeagent.workspace import Workspace, WorkspaceError


class _Pause(Exception):
    """Not the model's fault (server down, low resources, stop requested): retry later."""


class _Fail(Exception):
    """The step could not be completed."""


@dataclass
class StepResult:
    title: str
    status: str                       # done | failed | skipped | paused
    attempts: int = 0
    reason: str = ""
    commit: str | None = None


@dataclass
class RunReport:
    plan_title: str
    branch: str
    results: list[StepResult] = field(default_factory=list)

    @property
    def outcome(self) -> str:
        statuses = {r.status for r in self.results}
        if "failed" in statuses:
            return "failed"
        if "paused" in statuses:
            return "paused"
        return "done"

    def summary(self) -> str:
        lines = [f"Plan: {self.plan_title}  (branch {self.branch})  ->  {self.outcome.upper()}"]
        for r in self.results:
            extra = f"commit {r.commit}, {r.attempts} model calls" if r.status == "done" else r.reason
            lines.append(f"  [{r.status}] {r.title}" + (f"  ({extra})" if extra else ""))
        return "\n".join(lines)


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items)


class Runner:
    def __init__(self, cfg: Config, plan: Plan, ws: Workspace, llm: LLMClient, *,
                 guard=None, run_cmd=run_command, log: Callable[[str], None] | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 should_stop: Callable[[], bool] = lambda: False):
        first = shlex.split(plan.test_command)[0]
        if first not in cfg.allowed_commands:
            raise PlanError(f"test_command {first!r} is not in allowed_commands {cfg.allowed_commands}")
        self.cfg, self.plan, self.ws, self.llm = cfg, plan, ws, llm
        self.guard = guard or ResourceGuard(cfg, ws.root)
        self.run_cmd = run_cmd
        self.log = log or (lambda m: None)
        self.sleep = sleep
        self.should_stop = should_stop
        self.attempts = 0
        self.replies = ReplyLog(ws.root)

    # ------------------------------------------------------------------ run
    def run(self) -> RunReport:
        with WorkspaceLock(self.ws.root):
            git.ensure_repo(self.ws.root)
            branch = f"agent/{plan_slug(self.plan)}"
            git.checkout_branch(self.ws.root, branch)
            state = self.state = RunState(self.ws.root)
            report = RunReport(self.plan.title, branch)
            for step in self.plan.steps:
                if self.should_stop():
                    report.results.append(StepResult(step.title, "paused", reason="stop requested"))
                    break
                if state.is_done(self.plan, step):
                    report.results.append(StepResult(step.title, "skipped"))
                    continue
                self.log(f"== step: {step.title}")
                result = self._run_step(step)
                self.log(f"   -> {result.status} {result.reason}".rstrip())
                report.results.append(result)
                if result.status == "paused":
                    break
                if result.status == "failed" and self.cfg.stop_on_failure:
                    break
            return report

    # ------------------------------------------------------------------ one step
    def _run_step(self, step: Step) -> StepResult:
        paths = (*step.tests, *step.files)
        snapshot = self.ws.snapshot(paths)
        self.attempts = 0
        try:
            tests, satisfied, output = self._tests_phase(step)
            if satisfied:
                self.log("   implementation already satisfies the tests")
            else:
                self._impl_phase(step, tests, output)
            self._write(tests)                             # tests are exactly what we approved
            sha = git.commit_paths(self.ws.root, paths, step.commit)
            self.state.mark_done(self.plan, step, sha)
            return StepResult(step.title, "done", self.attempts, commit=sha)
        except _Pause as p:
            self.ws.restore(snapshot)
            return StepResult(step.title, "paused", self.attempts, reason=str(p))
        except (_Fail, git.GitError, WorkspaceError) as e:
            self.ws.restore(snapshot)
            return StepResult(step.title, "failed", self.attempts, reason=str(e))

    def _tests_phase(self, step: Step):
        feedback = None
        for n in range(1, self.cfg.max_attempts + 1):
            existing = self._read_many((*step.files, *step.context))
            text = self._ask(build_tests_prompt(self.plan, step, existing, self.cfg.max_context_chars, feedback),
                             step=step, phase="tests", attempt=n,
                             num_predict=min(self.cfg.num_predict_tests, self.cfg.num_predict))
            files, problems = self._collect(text, step.tests)
            if problems:
                feedback = _bullets(problems)
                self.log(f"   tests attempt {n}: {problems[0]}")
                continue
            self._write(files)
            res = self._run_tests()
            if res.timed_out or res.flooded:
                feedback = "Your tests hung or printed endlessly. Write simple, fast tests with no loops or input()."
                continue
            if res.ok:
                if all(self.ws.exists(f) for f in step.files):
                    return files, True, res.output
                feedback = ("These tests PASS even though the feature is not implemented yet, so they test "
                            "nothing. Assert concrete results from the description so they FAIL first.")
                self.log(f"   tests attempt {n}: vacuous (passed with no implementation)")
                continue
            return files, False, res.output                 # red: exactly what we want
        raise _Fail(f"could not get valid failing tests after {self.cfg.max_attempts} attempts: "
                    f"{(feedback or '').splitlines()[0] if feedback else 'unknown'}")

    def _impl_phase(self, step: Step, tests: dict[str, str], output: str) -> None:
        feedback = None
        for n in range(1, self.cfg.max_attempts + 1):
            existing = self._read_many((*step.files, *step.context))
            text = self._ask(build_impl_prompt(self.plan, step, tests, existing,
                                               self.cfg.max_context_chars, output, feedback),
                             step=step, phase="impl", attempt=n,
                             num_predict=self.cfg.num_predict)
            files, problems = self._collect(text, step.files)
            if problems:
                feedback = _bullets(problems)
                self.log(f"   impl attempt {n}: {problems[0]}")
                continue
            feedback = None
            self._write(files)
            self._write(tests)
            res = self._run_tests()
            if res.ok:
                return
            output = res.output
            self.log(f"   impl attempt {n}: tests still failing")
        tail = output.strip().splitlines()[-1] if output.strip() else ""
        raise _Fail(f"tests still failing after {self.cfg.max_attempts} attempts. Last line: {tail}")

    # ------------------------------------------------------------------ pieces
    def _retry_temperature(self, attempt: int) -> float:
        """Attempt 1 uses the configured temperature; each retry runs hotter. A small model that
        fell into a loop at a low temperature would otherwise repeat the exact same loop."""
        return min(self.cfg.temperature + 0.25 * (attempt - 1), 1.0)

    def _ask(self, messages: list[dict], *, step: Step, phase: str, attempt: int,
             num_predict: int | None = None) -> str:
        if self.should_stop():
            raise _Pause("stop requested")
        ok = self.guard.wait_until_ok(on_wait=lambda r: self.log("   waiting for resources: " + "; ".join(r)),
                                      should_stop=self.should_stop)
        if not ok:
            if self.should_stop():
                raise _Pause("stop requested")
            raise _Pause("system resources stayed too low; will retry later")
        temperature = self._retry_temperature(attempt)
        started = time.monotonic()
        try:
            self.attempts += 1
            text = self.llm.chat(messages, temperature=temperature, num_predict=num_predict)
        except LLMError as e:
            raise _Pause(f"model unavailable: {e}") from e
        seconds = time.monotonic() - started
        truncated = bool(getattr(text, "truncated", False))
        self.replies.save(step.title, phase, attempt, str(text), seconds=seconds, truncated=truncated,
                          temperature=temperature)
        self.log(f"   model replied in {seconds:.0f}s ({len(text)} chars)"
                 + (" - CUT OFF at the token limit" if truncated else ""))
        self.sleep(self.cfg.step_cooldown_s)              # let the laptop breathe
        return text

    def _collect(self, text: str, allowed: tuple[str, ...]) -> tuple[dict[str, str], list[str]]:
        parsed = parse_file_blocks(str(text), truncated=bool(getattr(text, "truncated", False)))
        for note in parsed.notes:
            self.log(f"   fixed up: {note}")
        problems = list(parsed.problems)
        if not parsed.blocks and not problems:
            return {}, ["your reply contained no FILE blocks; reply only with: FILE: <path> then a fenced code block"]
        got: dict[str, str] = {}
        ignored = []
        for b in parsed.blocks:
            (got.__setitem__(b.path, b.content) if b.path in allowed else ignored.append(b.path))
        if ignored:
            self.log(f"   ignored files outside this step: {', '.join(ignored)}")
        for path in allowed:
            if path in got:
                continue
            if path.endswith("__init__.py"):
                if not self.ws.exists(path):
                    got[path] = ""                          # trivial file: create it for the model
                continue
            problems.append(f"missing file: {path} (write every requested file, complete)")
        if ignored and problems:
            problems.insert(0, f"you wrote {', '.join(ignored)}, which this step does not allow. "
                               f"Write ONLY: {', '.join(allowed)}")
        for path, content in got.items():
            if len(content.encode("utf-8")) > self.cfg.max_file_bytes:
                problems.append(f"{path}: file is too large")
            elif path.endswith(".py"):
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    problems.append(f"{path}: syntax error on line {e.lineno}: {e.msg}")
        return got, problems

    def _write(self, files: dict[str, str]) -> None:
        for path, content in files.items():
            self.ws.write(path, content)

    def _read_many(self, paths) -> dict[str, str | None]:
        out = {}
        for p in paths:
            try:
                out[p] = self.ws.read(p)
            except WorkspaceError:
                out[p] = None
        return out

    def _run_tests(self) -> CommandResult:
        return self.run_cmd(self.plan.test_command, self.ws.root, allowed=self.cfg.allowed_commands,
                            timeout=self.cfg.test_timeout, mem_mb=self.cfg.command_mem_mb,
                            max_output_chars=self.cfg.max_output_chars)
