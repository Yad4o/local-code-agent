"""Git helpers for unattended commits: never prompts, never hangs, never runs
user hooks, and commits ONLY the files it is told to."""
import os
import subprocess
from pathlib import Path

GIT_TIMEOUT = 60
# kept out of `git status` via .git/info/exclude (never touches the user's .gitignore)
EXCLUDES = (".codeagent/", "__pycache__/", ".pytest_cache/")


class GitError(Exception):
    pass


def _run(root: str | Path, *args: str, check: bool = True) -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_PAGER": "cat"}
    try:
        p = subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True,
                           timeout=GIT_TIMEOUT, env=env, stdin=subprocess.DEVNULL)
    except FileNotFoundError as e:
        raise GitError("git is not installed or not on PATH") from e
    except (subprocess.TimeoutExpired, NotADirectoryError, OSError) as e:
        raise GitError(f"git {' '.join(args[:2])} failed: {e}") from e
    if check and p.returncode != 0:
        raise GitError(f"git {' '.join(args[:2])} failed: {(p.stderr or p.stdout).strip()}")
    return p.stdout.strip()


def ensure_repo(root: str | Path) -> bool:
    """Make `root` a git repo (branch main). True if it had to be created."""
    root = Path(root)
    created = not (root / ".git").exists()
    if created:
        _run(root, "init", "-q", "-b", "main")
    exclude = root / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    current = exclude.read_text() if exclude.exists() else ""
    have = set(current.splitlines())
    missing = [e for e in EXCLUDES if e not in have]
    if missing:
        sep = "" if current.endswith("\n") or not current else "\n"
        exclude.write_text(current + sep + "\n".join(missing) + "\n")
    return created


def current_branch(root: str | Path) -> str:
    return _run(root, "symbolic-ref", "--short", "HEAD")


def checkout_branch(root: str | Path, name: str) -> None:
    exists = subprocess.run(["git", "show-ref", "--verify", "--quiet", f"refs/heads/{name}"],
                            cwd=str(root)).returncode == 0
    _run(root, "checkout", "-q", name) if exists else _run(root, "checkout", "-q", "-b", name)


def _has_identity(root: str | Path) -> bool:
    return bool(_run(root, "config", "user.name", check=False)) and \
        bool(_run(root, "config", "user.email", check=False))


def commit_paths(root: str | Path, paths: list[str] | tuple[str, ...], message: str) -> str | None:
    """Commit exactly `paths`. Returns the short hash, or None if nothing changed."""
    _run(root, "add", "--", *paths)
    if not _run(root, "status", "--porcelain", "--", *paths):
        return None
    cfg = ["-c", "commit.gpgsign=false"]              # a signing prompt would hang forever
    if not _has_identity(root):
        cfg += ["-c", "user.name=codeagent", "-c", "user.email=codeagent@localhost"]
    _run(root, *cfg, "commit", "-q", "--no-verify", "-m", message, "--", *paths)
    return _run(root, "rev-parse", "--short", "HEAD")
