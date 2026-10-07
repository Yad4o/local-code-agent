"""24/7 mode: watch an inbox folder for plan files and build them one at a time.

inbox/        drop *.toml plans here
  running/    the plan being built (anything left here after a crash is re-queued)
  done/       finished plans + <name>.report.txt
  failed/     rejected/failed plans + .report.txt or .error.txt (with the reason)

The loop never dies from a bad plan: every plan is isolated, every error is
written down, and repeated outages back off instead of spinning the CPU.
"""
import shutil
import time
import traceback
from pathlib import Path
from typing import Callable

from codeagent.config import Config
from codeagent.llm import make_client
from codeagent.plan import PlanError, load_plan
from codeagent.resources import ResourceGuard
from codeagent.runner import Runner
from codeagent.state import WorkspaceBusy
from codeagent.workspace import Workspace

MAX_BACKOFF = 600.0


class Daemon:
    def __init__(self, cfg: Config, inbox: str | Path, *,
                 llm_factory: Callable = make_client,
                 guard_factory: Callable = lambda cfg, root: ResourceGuard(cfg, root),
                 log: Callable[[str], None] = print,
                 sleep: Callable[[float], None] = time.sleep,
                 should_stop: Callable[[], bool] = lambda: False,
                 settle_seconds: float = 2.0):
        self.cfg = cfg
        self.inbox = Path(inbox).expanduser()
        self.llm_factory = llm_factory
        self.guard_factory = guard_factory
        self.log = log
        self.sleep = sleep
        self.should_stop = should_stop
        self.settle_seconds = settle_seconds
        self.pause_delay = cfg.poll_seconds

    # ------------------------------------------------------------------ setup
    def prepare(self) -> None:
        for sub in ("running", "done", "failed"):
            (self.inbox / sub).mkdir(parents=True, exist_ok=True)
        for leftover in sorted((self.inbox / "running").glob("*.toml")):    # crash recovery
            dest = self._unique(self.inbox / leftover.name)
            shutil.move(str(leftover), dest)
            self.log(f"re-queued interrupted plan: {leftover.name}")

    def _queued(self) -> list[Path]:
        now = time.time()
        files = [p for p in self.inbox.glob("*.toml")
                 if p.is_file() and now - p.stat().st_mtime >= self.settle_seconds]
        return sorted(files, key=lambda p: (p.stat().st_mtime, p.name))

    @staticmethod
    def _unique(dest: Path) -> Path:
        if not dest.exists():
            return dest
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return dest.with_name(f"{dest.stem}.{stamp}{dest.suffix}")

    # ------------------------------------------------------------------ loop
    def run_once(self) -> int:
        """Process everything currently queued. Returns how many plans were handled."""
        handled = 0
        for path in self._queued():
            if self.should_stop():
                break
            outcome = self._process(path)
            handled += 1
            if outcome == "paused":
                self.pause_delay = min(self.pause_delay * 2, MAX_BACKOFF)
                break                                  # model/resources down: stop hammering
            if outcome == "busy":
                continue
            if outcome == "done":
                self.pause_delay = self.cfg.poll_seconds
        return handled

    def run_forever(self) -> None:
        self.prepare()
        self.log(f"watching {self.inbox} (Ctrl+C or SIGTERM to stop)")
        while not self.should_stop():
            self.run_once()
            self._nap(self.pause_delay)
        self.log("stopped")

    def _nap(self, seconds: float) -> None:
        """Sleep in 1s slices so a stop request is noticed quickly."""
        remaining = seconds
        while remaining > 0 and not self.should_stop():
            chunk = min(1.0, remaining)
            self.sleep(chunk)
            remaining -= chunk

    # ------------------------------------------------------------------ one plan
    def _process(self, path: Path) -> str:
        running = self.inbox / "running" / path.name
        shutil.move(str(path), running)
        self.log(f"plan: {path.name}")
        try:
            plan = load_plan(running)
            if not plan.workspace:
                raise PlanError("daemon mode needs a 'workspace = \"/path\"' line in the plan")
            ws = Workspace(Path(plan.workspace).expanduser(), self.cfg.max_file_bytes)
            runner = Runner(self.cfg, plan, ws, self.llm_factory(self.cfg),
                            guard=self.guard_factory(self.cfg, ws.root), log=self.log,
                            should_stop=self.should_stop)
            report = runner.run()
        except WorkspaceBusy as e:
            self.log(f"   busy: {e}; will retry")
            shutil.move(str(running), self._unique(self.inbox / path.name))
            return "busy"
        except PlanError as e:
            self._file(running, "failed", error=f"Plan rejected:\n{e}\n")
            return "failed"
        except Exception:                              # never let one plan kill the daemon
            self._file(running, "failed", error=traceback.format_exc())
            return "failed"

        text = report.summary() + "\n"
        if report.outcome == "done":
            self._file(running, "done", report=text)
        elif report.outcome == "failed":
            self._file(running, "failed", report=text)
        else:                                          # paused: try again later (progress is saved)
            self.log(f"   paused: {report.results[-1].reason}")
            shutil.move(str(running), self._unique(self.inbox / path.name))
        return report.outcome

    def _file(self, running: Path, bucket: str, *, report: str | None = None, error: str | None = None):
        dest = self._unique(self.inbox / bucket / running.name)
        shutil.move(str(running), dest)
        if report is not None:
            dest.with_suffix(".report.txt").write_text(report, encoding="utf-8")
        if error is not None:
            dest.with_suffix(".error.txt").write_text(error, encoding="utf-8")
        self.log(f"   filed under {bucket}/{dest.name}")
