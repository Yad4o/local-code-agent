"""Structured plan format (TOML). Strict on purpose: every plan looks the same,
and mistakes are reported all at once, before the model runs anything."""
import re
import shlex
import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PurePosixPath

TOP_KEYS = {"title", "test_command", "workspace", "step"}
STEP_KEYS = {"title", "description", "commit", "tests", "files", "context"}
FORBIDDEN_ROOTS = {".git", ".codeagent"}


class PlanError(Exception):
    def __init__(self, errors: list[str] | str):
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("\n".join(f"- {e}" for e in self.errors))


@dataclass(frozen=True)
class Step:
    title: str
    description: str
    commit: str
    tests: tuple[str, ...]
    files: tuple[str, ...]
    context: tuple[str, ...] = ()


@dataclass(frozen=True)
class Plan:
    title: str
    test_command: str
    workspace: str | None
    steps: tuple[Step, ...]


def template_text() -> str:
    return resources.files("codeagent").joinpath("plan_template.toml").read_text(encoding="utf-8")


def plan_slug(plan: Plan) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", plan.title.lower()).strip("-")
    return slug or "plan"


def _nonempty_str(d: dict, key: str, where: str, errors: list[str]) -> str | None:
    v = d.get(key)
    if not isinstance(v, str) or not v.strip():
        errors.append(f"{where}: '{key}' is required and must be a non-empty string")
        return None
    return v.strip()


def _path_list(d: dict, key: str, where: str, errors: list[str], required: bool) -> tuple[str, ...]:
    if key not in d:
        if required:
            errors.append(f"{where}: '{key}' is required (a non-empty list of file paths)")
        return ()
    v = d[key]
    if not isinstance(v, list) or (required and not v) or not all(isinstance(p, str) for p in v):
        errors.append(f"{where}: '{key}' must be a {'non-empty ' if required else ''}list of strings")
        return ()
    out, seen = [], set()
    for raw in v:
        bad = _bad_path(raw)
        if bad:
            errors.append(f"{where}: bad path in '{key}' ({raw!r}): {bad}")
            continue
        if raw in seen:
            errors.append(f"{where}: duplicate path in '{key}': {raw}")
            continue
        seen.add(raw)
        out.append(raw)
    return tuple(out)


def _bad_path(p: str) -> str | None:
    if not p.strip():
        return "empty"
    if "\\" in p:
        return "use forward slashes"
    pp = PurePosixPath(p)
    if pp.is_absolute():
        return "must be relative to the workspace"
    if ".." in pp.parts:
        return "must not contain '..'"
    if pp.parts and pp.parts[0] in FORBIDDEN_ROOTS:
        return f"{pp.parts[0]}/ is reserved"
    return None


def parse_plan(text: str) -> Plan:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise PlanError(f"invalid TOML: {e}") from e

    errors: list[str] = []
    for k in sorted(set(data) - TOP_KEYS):
        errors.append(f"unknown top-level key '{k}'")
    title = _nonempty_str(data, "title", "plan", errors)
    test_command = _nonempty_str(data, "test_command", "plan", errors)
    if test_command:
        try:
            shlex.split(test_command)
        except ValueError as e:
            errors.append(f"plan: 'test_command' is not valid shell syntax: {e}")
    workspace = data.get("workspace")
    if workspace is not None and (not isinstance(workspace, str) or not workspace.strip()):
        errors.append("plan: 'workspace' must be a non-empty string when present")
        workspace = None

    raw_steps = data.get("step")
    if not isinstance(raw_steps, list) or not raw_steps:
        errors.append("plan: at least one [[step]] is required")
        raw_steps = []

    steps, titles = [], set()
    for i, rs in enumerate(raw_steps, 1):
        where = f"step {i}"
        if not isinstance(rs, dict):
            errors.append(f"{where}: must be a table")
            continue
        for k in sorted(set(rs) - STEP_KEYS):
            errors.append(f"{where}: unknown key '{k}'")
        st = _nonempty_str(rs, "title", where, errors)
        if st in titles:
            errors.append(f"{where}: duplicate step title '{st}'")
        titles.add(st)
        desc = _nonempty_str(rs, "description", where, errors)
        commit = _nonempty_str(rs, "commit", where, errors)
        if commit and ("\n" in commit or len(commit) > 72):
            errors.append(f"{where}: 'commit' must be a single line of at most 72 characters")
        tests = _path_list(rs, "tests", where, errors, required=True)
        files = _path_list(rs, "files", where, errors, required=True)
        context = _path_list(rs, "context", where, errors, required=False)
        both = set(tests) & set(files)
        if both:
            errors.append(f"{where}: path(s) listed in both 'tests' and 'files': {', '.join(sorted(both))}")
        if st and desc and commit:
            steps.append(Step(st, desc, commit, tests, files, context))

    if errors:
        raise PlanError(errors)
    return Plan(title, test_command, workspace, tuple(steps))


def load_plan(path: str | Path) -> Plan:
    p = Path(path)
    if not p.is_file():
        raise PlanError(f"plan file not found: {p}")
    return parse_plan(p.read_text(encoding="utf-8"))
