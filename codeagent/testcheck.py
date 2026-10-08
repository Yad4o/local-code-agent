"""Catch broken test files before they poison a whole step.

A test that can never run (a parameter nothing provides, a syntax error, no tests
at all) fails every implementation the same way, so retrying the implementation
is pointless. These checks let the runner send the problem back to the test phase.
"""
import ast
import re

BUILTIN_FIXTURES = frozenset({
    "tmp_path", "tmp_path_factory", "tmpdir", "tmpdir_factory", "monkeypatch", "capsys",
    "capsysbinary", "capfd", "capfdbinary", "caplog", "request", "recwarn", "pytestconfig",
    "cache", "record_property", "record_testsuite_property", "doctest_namespace",
})


def _is_decorator(node: ast.expr, name: str) -> bool:
    target = node.func if isinstance(node, ast.Call) else node
    return (isinstance(target, ast.Attribute) and target.attr == name) or \
           (isinstance(target, ast.Name) and target.id == name)


def _parametrized_names(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for dec in func.decorator_list:
        if not (isinstance(dec, ast.Call) and _is_decorator(dec, "parametrize") and dec.args):
            continue
        first = dec.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            names.update(n.strip() for n in first.value.split(","))
        elif isinstance(first, (ast.List, ast.Tuple)):
            names.update(e.value for e in first.elts
                         if isinstance(e, ast.Constant) and isinstance(e.value, str))
    return names


def _required_params(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Parameters pytest must supply: the ones without a default value."""
    a = func.args
    positional = [*a.posonlyargs, *a.args]
    required = [p.arg for p in positional[:len(positional) - len(a.defaults)]]
    required += [p.arg for p, d in zip(a.kwonlyargs, a.kw_defaults) if d is None]
    return [p for p in required if p not in ("self", "cls")]


def check_test_source(source: str, known_fixtures: frozenset[str] | set[str] = frozenset()) -> list[str]:
    """Problems that make a test file unrunnable. Syntax errors are reported elsewhere."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    defined = {f.name for f in funcs if any(_is_decorator(d, "fixture") for d in f.decorator_list)}
    available = BUILTIN_FIXTURES | defined | set(known_fixtures)
    tests = [f for f in funcs if f.name.startswith("test") and f.name not in defined]
    if not tests:
        return ["there is no test function; write functions named test_something()"]
    problems = []
    for scope in [tree, *[n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]]:
        seen: set[str] = set()
        for f in scope.body:
            if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name.startswith("test"):
                if f.name in seen:
                    problems.append(f"{f.name}() is defined twice; the second silently replaces the first. "
                                    f"Give every test its own name.")
                seen.add(f.name)
    for f in tests:
        missing = [p for p in _required_params(f) if p not in available | _parametrized_names(f)]
        if missing:
            problems.append(
                f"{f.name}() has parameter(s) {', '.join(missing)} that nothing provides, so pytest "
                f"errors before running it. Every test function must take NO parameters: "
                f"write the input value inside the test.")
    return problems


def diagnose_broken_tests(output: str) -> str | None:
    """Spot pytest output that is the tests' fault, not the implementation's."""
    m = re.search(r"fixture '([^']+)' not found", output)
    if m:
        return (f"pytest could not find fixture '{m.group(1)}'. Every test function must take NO "
                f"parameters: write the input value inside the test.")
    if re.search(r"^E\s+(SyntaxError|IndentationError|TabError)\b", output, re.M):
        return "the test file has a syntax error; write plain, valid Python."
    if re.search(r"\bno tests ran\b", output):
        return "pytest found no tests; name each test function starting with test_."
    return None
