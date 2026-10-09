"""Turn the human's examples into a test file, with no model involved.

The tiny model is an unreliable typist: when it also writes the tests it invents
contradictory expectations, calls the function with the wrong arguments and repeats
test names. So the plan carries the facts and this module prints them as pytest code.
"""
import ast

from codeagent.plan import Example, Step


def _call(name: str, ex: Example) -> str:
    parts = [repr(a) for a in ex.args] + [f"{k}={v!r}" for k, v in (ex.kwargs or {}).items()]
    return f"{name}({', '.join(parts)})"


def render_tests(step: Step) -> str:
    module, name = step.function.split(":")
    lines = []
    if any(e.raises for e in step.examples):
        lines += ["import pytest"]
    lines += [f"from {module} import {name}", "", ""]
    for i, ex in enumerate(step.examples, 1):
        if ex.raises:
            lines += [f"def test_example_{i}_raises_{ex.raises}():",
                      f"    with pytest.raises({ex.raises}):",
                      f"        {_call(name, ex)}"]
        else:
            lines += [f"def test_example_{i}():", f"    assert {_call(name, ex)} == {ex.expect!r}"]
        lines += ["", ""]
    source = "\n".join(lines).rstrip("\n") + "\n"
    ast.parse(source)                                   # a bug here must be loud, never silent
    return source


def defined_names(source: str) -> set[str]:
    """Names defined at the top level of a module: def, class, assignment, import."""
    names: set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return names
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(t.id for t in targets if isinstance(t, ast.Name))
    return names


def returns_a_value(source: str, name: str) -> bool | None:
    """Does function `name` have a path that returns (or yields) something? None = cannot tell."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    fn = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name), None)
    if fn is None:
        return None
    stack = list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue                                    # a nested function's return is not ours
        if isinstance(node, ast.Return) and node.value is not None \
                and not (isinstance(node.value, ast.Constant) and node.value.value is None):
            return True
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            return True
        stack.extend(ast.iter_child_nodes(node))
    return False
