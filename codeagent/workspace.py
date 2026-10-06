"""Sandboxed file access. The agent can only touch files inside the workspace
root, never .git/ or .codeagent/, never through symlinks that leave the root."""
import os
import tempfile
from pathlib import Path, PurePosixPath

RESERVED = {".git", ".codeagent"}


class WorkspaceError(Exception):
    pass


class Workspace:
    def __init__(self, root: str | Path, max_file_bytes: int):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_file_bytes = max_file_bytes

    def resolve(self, rel: str) -> Path:
        if not rel or not rel.strip():
            raise WorkspaceError("empty path")
        pp = PurePosixPath(rel)
        if pp.is_absolute():
            raise WorkspaceError(f"absolute path not allowed: {rel}")
        if ".." in pp.parts:
            raise WorkspaceError(f"'..' not allowed in path: {rel}")
        if pp.parts[0] in RESERVED:
            raise WorkspaceError(f"{pp.parts[0]}/ is reserved: {rel}")
        full = (self.root / pp).resolve()
        if not full.is_relative_to(self.root):
            raise WorkspaceError(f"path resolves outside the workspace: {rel}")
        return full

    def exists(self, rel: str) -> bool:
        return self.resolve(rel).is_file()

    def read(self, rel: str) -> str | None:
        p = self.resolve(rel)
        if not p.is_file():
            return None
        if p.stat().st_size > self.max_file_bytes:
            raise WorkspaceError(f"file too large to read: {rel}")
        try:
            return p.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise WorkspaceError(f"file is not valid UTF-8 text: {rel}") from e

    def write(self, rel: str, content: str) -> None:
        p = self.resolve(rel)
        if len(content.encode("utf-8")) > self.max_file_bytes:
            raise WorkspaceError(f"file too large to write: {rel}")
        p.parent.mkdir(parents=True, exist_ok=True)
        # atomic: a crash mid-write can never leave a half-written source file
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".tmp-", suffix=".part")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                f.write(content)
            os.replace(tmp, p)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def delete(self, rel: str) -> None:
        self.resolve(rel).unlink(missing_ok=True)

    def snapshot(self, rels: list[str] | tuple[str, ...]) -> dict[str, str | None]:
        return {r: self.read(r) for r in rels}

    def restore(self, snap: dict[str, str | None]) -> None:
        for rel, content in snap.items():
            if content is None:
                self.delete(rel)
            else:
                self.write(rel, content)
