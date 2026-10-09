"""Skeleton mode: the plan gives the exact signature, the harness writes everything except the body.

The model replies with a few body lines instead of a whole file, so there is far less to get wrong
(no repeated headers, no re-declared imports, no wrong function name) and far less to wait for.
"""
import ast
import re
import textwrap

from codeagent.plan import Step

BODY_STOP = ["\nFILE:"]               # the model has nothing else to write after the body


def module_path(step: Step) -> str:
    return step.function.split(":")[0].replace(".", "/") + ".py"


def _function_name(step: Step) -> str:
    return step.function.split(":")[1]


def signature_problem(signature: str, name: str) -> str | None:
    try:
        tree = ast.parse(f"def {signature}:\n    pass\n")
    except SyntaxError as e:
        return f"is not a valid function signature ({e.msg})"
    fn = tree.body[0]
    if not isinstance(fn, ast.FunctionDef) or fn.name != name:
        return f"must define the function '{name}'"
    return None


def init_source(step: Step, init_path: str) -> str:
    """`pkg/__init__.py` re-exports the function so `from pkg import name` works."""
    pkg = init_path[: -len("/__init__.py")].replace("/", ".") if "/" in init_path else ""
    module = step.function.split(":")[0]
    rest = module[len(pkg) + 1:] if pkg else module
    return f"from .{rest} import {_function_name(step)}\n"


def assemble_module(step: Step, body: str) -> tuple[str | None, str | None]:
    """(source, None) or (None, problem). Accepts a bare body; also a whole `def` if the model wrote one."""
    name = _function_name(step)
    text = textwrap.dedent(body).strip("\n")
    if not text.strip():
        return None, "the function body is empty; write the lines that go inside the function"
    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = None
    if tree and any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name for n in tree.body):
        return text + "\n", None
    lines = text.splitlines()
    i = 0
    while i < len(lines) and re.match(r"(import|from)\s+\S", lines[i]):
        i += 1                                           # leading imports belong at the top of the module
    imports, rest = lines[:i], lines[i:]
    src = "\n".join([*imports, "", f"def {step.signature}:", *[("    " + l) if l.strip() else "" for l in rest], ""])
    try:
        ast.parse(src)
    except SyntaxError as e:
        return None, f"the body is not valid Python inside `def {step.signature}:` ({e.msg}, line {e.lineno} of the module)"
    return src, None


def assemble_files(step: Step, got: dict[str, str],
                   existing_inits: dict[str, str | None] | None = None) -> tuple[dict[str, str], list[str]]:
    """Turn the model's body into the complete set of files the step owns. A package `__init__.py` that
    earlier steps already filled keeps its lines; this step's re-export is added to it."""
    path = module_path(step)
    source, problem = assemble_module(step, got.get(path, ""))
    if problem:
        return {}, [problem]
    files = {path: source}
    for p in step.files:
        if p.endswith("__init__.py") and p != path:
            line = init_source(step, p)
            old = (existing_inits or {}).get(p) or ""
            files[p] = old if line.strip() in old.splitlines() else old + ("" if not old or old.endswith("\n") else "\n") + line
    return files, []
