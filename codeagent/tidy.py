"""Deterministic clean-up tools. The model writes the logic; code does the tidying.

* `tidy_python`: removes docstrings and comments and re-prints the code in one standard style
  (stdlib `ast`, no dependencies). A tiny model's chatter never reaches your files.
* `ruff_fix` / `ruff_problems`: optional (used only when `ruff` is installed). Removes unused imports
  and reports real mistakes (undefined names, redefinitions, syntax errors) with exact line numbers.
"""
import ast
import shutil
import subprocess
from pathlib import Path

RUFF_FIX_RULES = "F401"                          # unused imports
RUFF_CHECK_RULES = "F821,F811,E9"                # undefined name, redefinition, syntax/IO errors


def _drop_docstring(body: list[ast.stmt]) -> None:
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        del body[0]


def tidy_python(source: str) -> str:
    """Docstrings and comments out, canonical formatting in. Returns the source unchanged if it
    does not parse (the caller's own checks report that)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _drop_docstring(node.body)
            if not node.body and not isinstance(node, ast.Module):
                node.body = [ast.Pass()]
    out = ast.unparse(ast.fix_missing_locations(tree))
    return out + "\n" if out else ""


def ruff_available() -> bool:
    return shutil.which("ruff") is not None


def _ruff(args: list[str], root: Path, timeout: float) -> str | None:
    try:
        proc = subprocess.run(["ruff", *args], cwd=str(root), capture_output=True, text=True,
                              timeout=timeout, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout


def _code_paths(paths: list[str]) -> list[str]:
    return [p for p in paths if p.endswith(".py")]


def ruff_fix(root: Path, paths: list[str], timeout: float = 20.0) -> None:
    """Remove unused imports. `__init__.py` is skipped: its imports are re-exports on purpose."""
    todo = [p for p in _code_paths(paths) if not p.endswith("__init__.py")]
    if todo and ruff_available():
        _ruff(["check", "--fix", "--select", RUFF_FIX_RULES, "--no-cache", "--quiet", *todo], root, timeout)


def ruff_problems(root: Path, paths: list[str], timeout: float = 20.0) -> list[str]:
    """Real mistakes found without running anything, one short line each."""
    todo = _code_paths(paths)
    if not todo or not ruff_available():
        return []
    out = _ruff(["check", "--select", RUFF_CHECK_RULES, "--output-format", "concise",
                 "--no-cache", "--quiet", *todo], root, timeout)
    return [line.strip() for line in (out or "").splitlines() if line.strip() and ": " in line]
