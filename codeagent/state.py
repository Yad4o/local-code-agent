"""Crash-safe progress tracking + a lock so two agents never edit one workspace."""
import fcntl
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from codeagent.plan import Plan, Step

DIR = ".codeagent"


class WorkspaceBusy(Exception):
    pass


def fingerprint(step: Step) -> str:
    """Changes whenever the step's definition changes, so edited steps are re-run."""
    return hashlib.sha256(json.dumps(asdict(step), sort_keys=True).encode()).hexdigest()[:16]


class RunState:
    def __init__(self, root: str | Path):
        self.dir = Path(root) / DIR
        self.path = self.dir / "state.json"
        self.data = self._load()

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("plans"), dict):
                return data
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            try:                                   # keep the broken file for inspection
                self.path.replace(self.path.with_name("state.json.corrupt"))
            except OSError:
                pass
        return {"version": 1, "plans": {}}

    def is_done(self, plan: Plan, step: Step) -> bool:
        rec = self.data["plans"].get(plan.title, {}).get(step.title)
        return bool(rec) and rec.get("fingerprint") == fingerprint(step)

    def mark_done(self, plan: Plan, step: Step, commit: str | None) -> None:
        self.data["plans"].setdefault(plan.title, {})[step.title] = {
            "fingerprint": fingerprint(step), "commit": commit,
        }
        self._save()

    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=".state-", suffix=".part")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())               # survive a power cut mid-run
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


class WorkspaceLock:
    """Exclusive, non-blocking lock. Released automatically if the process dies."""

    def __init__(self, root: str | Path):
        self.path = Path(root) / DIR / "lock"
        self.fd: int | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self.fd)
            self.fd = None
            raise WorkspaceBusy(f"another codeagent is already working in {self.path.parent.parent}")
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None
