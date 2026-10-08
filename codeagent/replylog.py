"""Keep every raw model reply on disk (newest N), so you can read exactly what
the model wrote when a step fails. Writing here must never crash the agent."""
import re
import time
from pathlib import Path


class ReplyLog:
    def __init__(self, root: str | Path, keep: int = 60):
        self.dir = Path(root) / ".codeagent" / "replies"
        self.keep = keep
        self.counter = 0

    def save(self, step: str, phase: str, attempt: int, text: str, *, seconds: float,
             truncated: bool, temperature: float) -> Path | None:
        slug = re.sub(r"[^a-z0-9]+", "-", step.lower()).strip("-")[:40] or "step"
        self.counter += 1
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{self.counter:03d}-{slug}-{phase}-a{attempt}.txt"
        header = (f"# step: {step}\n# phase: {phase}\n# attempt: {attempt}\n"
                  f"# seconds: {seconds:.1f}\n# truncated: {'yes' if truncated else 'no'}\n"
                  f"# temperature: {temperature:g}\n---- raw reply ----\n")
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            path = self.dir / name
            path.write_text(header + text.rstrip("\n") + "\n", encoding="utf-8")
            for old in sorted(self.dir.glob("*.txt"))[:-self.keep]:
                old.unlink(missing_ok=True)
            return path
        except OSError:
            return None
