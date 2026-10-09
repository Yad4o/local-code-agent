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
from codeagent.parsing import FileBlock, parse_file_blocks
from codeagent.plan import Plan, PlanError, Step, plan_slug
from codeagent.prompts import build_impl_prompt, build_tests_prompt
from codeagent.replylog import ReplyLog
from codeagent.resources import ResourceGuard
from codeagent.shell import CommandResult, run_command
from codeagent.skeleton import BODY_STOP, assemble_files, module_path
from codeagent.spec_tests import defined_names, render_tests, returns_a_value
from codeagent.tidy import ruff_fix, ruff_problems, tidy_python
from codeagent.state import RunState, WorkspaceLock
from codeagent.testcheck import check_test_source, diagnose_broken_tests
from codeagent.workspace import Workspace, WorkspaceError


class _Pause(Exception):
    """Not the model's fault (server down, low resources, stop requested): retry later."""


class _Fail(Exception):
    """The step could not be completed."""


class _TestsBroken(Exception):
    """The tests themselves cannot run (not an implementation problem): rewrite them."""


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
            feedback = None
            for rewrite in range(self.cfg.max_test_rewrites + 1):
                tests, satisfied, output = self._tests_phase(step, feedback)
                if satisfied:
                    self.log("   implementation already satisfies the tests")
                    break
                try:
                    self._impl_phase(step, tests, output)
                    break
                except _TestsBroken as broken:
                    if step.examples:                       # generated from the plan: rewriting cannot help
                        raise _Fail(f"the tests generated from the plan's examples cannot run: {broken}") from broken
                    if rewrite == self.cfg.max_test_rewrites:
                        raise _Fail(f"the tests are broken and rewriting them did not help: {broken}") from broken
                    self.log(f"   the tests cannot run ({broken}); rewriting the tests")
                    self.ws.restore(snapshot)               # drop the half-built implementation too
                    feedback = str(broken)
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

    def _spec_tests_phase(self, step: Step):
        """The plan's examples ARE the tests: printed by the harness, never written by the model."""
        files = {step.tests[0]: render_tests(step)}
        self.log(f"   tests generated from {len(step.examples)} example(s) in the plan (no model call)")
        self._write(files)
        res = self._run_tests()
        if res.timed_out or res.flooded:
            raise _Fail("the tests generated from the plan's examples hung or printed endlessly; check the examples")
        if res.ok:
            if all(self.ws.exists(f) for f in step.files):
                return files, True, res.output
            raise _Fail("the plan's examples already pass with no implementation; check the examples")
        return files, False, res.output

    def _tests_phase(self, step: Step, feedback: str | None = None):
        if step.examples:
            return self._spec_tests_phase(step)
        for n in range(1, self.cfg.max_attempts + 1):
            existing = self._read_many((*step.files, *step.context))
            text = self._ask(build_tests_prompt(self.plan, step, existing, self.cfg.max_context_chars, feedback),
                             step=step, phase="tests", attempt=n,
                             num_predict=min(self.cfg.num_predict_tests, self.cfg.num_predict))
            files, problems = self._collect(text, step.tests)
            problems += self._lint_tests(files)
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
            broken = diagnose_broken_tests(res.output)
            if broken:                                      # red for the wrong reason: the tests are the problem
                feedback = broken
                self.log(f"   tests attempt {n}: tests cannot run ({broken})")
                continue
            return files, False, res.output                 # red: exactly what we want
        raise _Fail(f"could not get valid failing tests after {self.cfg.max_attempts} attempts: "
                    f"{(feedback or '').splitlines()[0] if feedback else 'unknown'}")

    def _impl_phase(self, step: Step, tests: dict[str, str], output: str) -> None:
        feedback = None
        seen: dict[str, int] = {}                        # code we already tried -> attempt number
        fresh = False                                    # next prompt hides the previous code (anti-anchoring)
        repeats = 0                                      # how many times the model repeated itself
        for n in range(1, self.cfg.max_attempts + 1):
            existing = {p: None for p in (*step.files, *step.context)} if fresh \
                else self._read_many((*step.files, *step.context))

            text = self._ask(build_impl_prompt(self.plan, step, tests, existing,
                                               self.cfg.max_context_chars, output, feedback),
                             step=step, phase="impl", attempt=n,
                             num_predict=self.cfg.num_predict_body if step.signature else self.cfg.num_predict,
                             stop=BODY_STOP if step.signature else None, bump=0.15 * repeats)
            fresh = False
            if step.signature:                       # skeleton mode: the model wrote only the body
                files, problems = self._collect(text, (module_path(step),), check_syntax=False)  # a body is not a module
                if not problems:
                    files, problems = assemble_files(step, files)
            else:
                files, problems = self._collect(text, step.files)
            if not problems:
                problems += self._lint_impl(step, files)
            if not problems and self.cfg.tidy_code:
                files = {p: tidy_python(c) if p.endswith(".py") else c for p, c in files.items()}
            if problems:
                feedback = _bullets(problems)
                self.log(f"   impl attempt {n}: {problems[0]}")
                continue
            key = "\n".join(f"{p}\n{files[p]}" for p in sorted(files))
            if key in seen:                              # a tiny model often copies its last answer verbatim
                repeats += 1
                fresh = True
                feedback = (f"this is identical to your attempt {seen[key]}, which already failed the tests. "
                            f"Write a DIFFERENT solution; do not copy your earlier code.")
                self.log(f"   impl attempt {n}: same code as attempt {seen[key]}; not re-running, asking for a different one")
                continue
            seen[key] = n
            self._write(files)
            self._write(tests)
            problems = self._static_check(list(files))
            if problems:
                feedback = _bullets(problems)
                self.log(f"   impl attempt {n}: static check: {problems[0]}")
                continue
            feedback = None
            res = self._run_tests()
            if res.ok:
                return
            broken = diagnose_broken_tests(res.output)
            if broken:
                raise _TestsBroken(broken)
            output = res.output
            self.log(f"   impl attempt {n}: tests still failing")
        tail = output.strip().splitlines()[-1] if output.strip() else ""
        raise _Fail(f"tests still failing after {self.cfg.max_attempts} attempts. Last line: {tail}")

    # ------------------------------------------------------------------ pieces
    def _retry_temperature(self, attempt: int, bump: float = 0.0) -> float:
        """Attempt 1 uses the configured temperature; retries warm up only a little, and never past
        the cap. A tiny model gets much worse when hot (it starts writing nonsense)."""
        ceiling = max(self.cfg.temperature, self.cfg.max_retry_temperature)
        t = min(self.cfg.temperature + self.cfg.retry_temperature_step * (attempt - 1), ceiling)
        return min(t + bump, max(ceiling, 0.7)) if bump else t     # repeating itself earns a warmer retry

    def _static_check(self, paths: list[str]) -> list[str]:
        """ruff, if installed: remove unused imports, then report undefined names before running anything."""
        if not self.cfg.use_ruff:
            return []
        ruff_fix(self.ws.root, paths)
        return ruff_problems(self.ws.root, paths)

    def _lint_impl(self, step: Step, files: dict[str, str]) -> list[str]:
        """A tool, not a prompt: the function the examples call must really be defined."""
        if not step.function:
            return []
        module, name = step.function.split(":")
        path = module.replace(".", "/") + ".py"
        if path not in files:
            return []
        if name not in defined_names(files[path]):
            return [f"{path} does not define `{name}`; write `def {name}(...)` at the top level of {path}"]
        if any(not e.raises for e in step.examples) and returns_a_value(files[path], name) is False:
            return [f"`{name}` never returns a value, but the examples expect it to return a result; "
                    f"end it with `return <the result>`"]
        return []

    def _lint_tests(self, files: dict[str, str]) -> list[str]:
        fixtures = set()
        try:
            conftest = self.ws.read("tests/conftest.py") or ""
            fixtures = {n.name for n in ast.walk(ast.parse(conftest)) if isinstance(n, ast.FunctionDef)}
        except (WorkspaceError, SyntaxError):
            pass
        out = []
        for path, content in files.items():
            if path.endswith(".py"):
                out += [f"{path}: {p}" for p in check_test_source(content, fixtures)]
        return out

    def _ask(self, messages: list[dict], *, step: Step, phase: str, attempt: int,
             num_predict: int | None = None, stop: list[str] | None = None, bump: float = 0.0) -> str:
        if self.should_stop():
            raise _Pause("stop requested")
        ok = self.guard.wait_until_ok(on_wait=lambda r: self.log("   waiting for resources: " + "; ".join(r)),
                                      should_stop=self.should_stop)
        if not ok:
            if self.should_stop():
                raise _Pause("stop requested")
            raise _Pause("system resources stayed too low; will retry later")
        temperature = self._retry_temperature(attempt, bump)
        started = time.monotonic()
        try:
            self.attempts += 1
            extra = {"stop": stop} if stop else {}
            text = self.llm.chat(messages, temperature=temperature, num_predict=num_predict, **extra)
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

    def _collect(self, text: str, allowed: tuple[str, ...], *,
                 check_syntax: bool = True) -> tuple[dict[str, str], list[str]]:
        parsed = parse_file_blocks(str(text), truncated=bool(getattr(text, "truncated", False)))
        for note in parsed.notes:
            self.log(f"   fixed up: {note}")
        problems = list(parsed.problems)
        targets = [p for p in allowed if not p.endswith("__init__.py")]
        if not parsed.blocks and len(parsed.anonymous) == 1 and len(targets) == 1:
            parsed.blocks = [FileBlock(targets[0], parsed.anonymous[0])]     # only one file is possible: no header needed
            self.log(f"   fixed up: the reply had no FILE header; used its only code block as {targets[0]}")
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
            elif check_syntax and path.endswith(".py"):
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
