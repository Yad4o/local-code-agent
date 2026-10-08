"""Structured plan format (TOML). Strict on purpose: every plan looks the same,
and mistakes are reported all at once, before the model runs anything."""
import builtins
import math
import re
import shlex
import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PurePosixPath

TOP_KEYS = {"title", "test_command", "workspace", "step"}
STEP_KEYS = {"title", "description", "commit", "tests", "files", "context", "function", "examples"}
EXAMPLE_KEYS = {"args", "kwargs", "expect", "raises"}
FUNCTION_RE = re.compile(r"^([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*):([A-Za-z_]\w*)$")
FORBIDDEN_ROOTS = {".git", ".codeagent"}


class PlanError(Exception):
    def __init__(self, errors: list[str] | str):
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("\n".join(f"- {e}" for e in self.errors))


@dataclass(frozen=True, eq=False)
class Example:
    """One input -> output fact about the function, written by the human in the plan.
    The harness turns it into a real test; the model never gets to decide what is correct."""
    args: tuple = ()
    kwargs: dict | None = None
    expect: object = None
    raises: str | None = None           # name of a builtin exception; None means "returns `expect`"


@dataclass(frozen=True, eq=False)
class Step:
    title: str
    description: str
    commit: str
    tests: tuple[str, ...]
    files: tuple[str, ...]
    context: tuple[str, ...] = ()
    function: str | None = None         # "package.module:name": the function the examples call
    examples: tuple[Example, ...] = ()


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


def _literal_problem(value, path: str = "value") -> str | None:
    """Only plain TOML data that prints as valid Python (no nan/inf, no dates)."""
    if isinstance(value, bool) or isinstance(value, (int, str)):
        return None
    if isinstance(value, float):
        return None if math.isfinite(value) else f"{path} is nan/inf, which cannot be written as a Python literal"
    if isinstance(value, list):
        return next((p for i, v in enumerate(value) if (p := _literal_problem(v, f"{path}[{i}]"))), None)
    if isinstance(value, dict):
        return next((p for k, v in value.items() if (p := _literal_problem(v, f"{path}[{k!r}]"))), None)
    return f"{path} has an unsupported type ({type(value).__name__}); use strings, numbers, booleans, lists, tables"


def _parse_examples(rs: dict, where: str, tests: tuple[str, ...], files: tuple[str, ...],
                    errors: list[str]) -> tuple[str | None, tuple[Example, ...]]:
    function, raw = rs.get("function"), rs.get("examples")
    if raw is None and function is None:
        return None, ()
    if raw is None or function is None:
        errors.append(f"{where}: 'function' and 'examples' go together (give both or neither)")
        return None, ()
    ok = True
    m = FUNCTION_RE.match(function) if isinstance(function, str) else None
    if not m:
        errors.append(f"{where}: 'function' must look like 'package.module:name', got {function!r}")
        ok = False
    elif m.group(1).replace(".", "/") + ".py" not in files:
        errors.append(f"{where}: 'function' module file {m.group(1).replace('.', '/')}.py must be listed in 'files'")
        ok = False
    if len(tests) != 1:
        errors.append(f"{where}: a step with examples needs exactly one path in 'tests' (the generated test file)")
        ok = False
    if not isinstance(raw, list) or not raw or not all(isinstance(e, dict) for e in raw):
        errors.append(f"{where}: 'examples' must be a non-empty list of tables ([[step.examples]])")
        return None, ()
    out = []
    for j, e in enumerate(raw, 1):
        ew = f"{where}, example {j}"
        for k in sorted(set(e) - EXAMPLE_KEYS):
            errors.append(f"{ew}: unknown key '{k}'")
        args, kwargs = e.get("args", []), e.get("kwargs", {})
        if not isinstance(args, list):
            errors.append(f"{ew}: 'args' must be a list")
            ok = False
        if not isinstance(kwargs, dict) or not all(k.isidentifier() for k in kwargs):
            errors.append(f"{ew}: 'kwargs' must be a table whose keys are valid parameter names")
            ok = False
        if ("expect" in e) == ("raises" in e):
            errors.append(f"{ew}: give exactly one of 'expect' (the result) or 'raises' (an exception name)")
            ok = False
        raises = e.get("raises")
        if raises is not None:
            exc = getattr(builtins, raises, None) if isinstance(raises, str) else None
            if not (isinstance(exc, type) and issubclass(exc, BaseException)):
                errors.append(f"{ew}: 'raises' must be a builtin exception name like ValueError, got {raises!r}")
                ok = False
        for key in ("args", "kwargs", "expect"):
            if key in e and (problem := _literal_problem(e[key], key)):
                errors.append(f"{ew}: {problem}")
                ok = False
        if ok:
            out.append(Example(tuple(args), dict(kwargs) or None, e.get("expect"), raises))
    return (function, tuple(out)) if ok else (None, ())


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
        function, examples = _parse_examples(rs, where, tests, files, errors)
        if st and desc and commit:
            steps.append(Step(st, desc, commit, tests, files, context, function, examples))

    if errors:
        raise PlanError(errors)
    return Plan(title, test_command, workspace, tuple(steps))


def load_plan(path: str | Path) -> Plan:
    p = Path(path)
    if not p.is_file():
        raise PlanError(f"plan file not found: {p}")
    return parse_plan(p.read_text(encoding="utf-8"))
