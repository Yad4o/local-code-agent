import json

import pytest

from agent import (
    PlanError,
    assemble_files,
    assemble_module,
    init_source,
    module_path,
    parse_plan,
    signature_problem,
)


def plan(**step_changes):
    step = {
        "title": "add", "description": "add_item", "commit": "feat: add",
        "tests": ["tests/test_add.py"], "files": ["cart/__init__.py", "cart/engine.py"],
        "function": "cart.engine:add_item", "signature": "add_item(cart, name, price, quantity=1)",
        "examples": [{"args": [[], "apple", 1.5], "expect": [{"name": "apple", "price": 1.5, "quantity": 1}]}],
    }
    step.update(step_changes)
    step = {k: v for k, v in step.items() if v is not None}
    return json.dumps({"title": "Cart", "test_command": "python -m pytest -q", "steps": [step]})


def step():
    return parse_plan(plan()).steps[0]


def test_signature_is_parsed_onto_the_step():
    assert step().signature == "add_item(cart, name, price, quantity=1)"
    assert module_path(step()) == "cart/engine.py"


def test_signature_must_be_valid_and_match_the_function_name():
    assert signature_problem("add_item(a, b=1)", "add_item") is None
    assert signature_problem("add_item(a, b=1) -> int", "add_item") is None
    assert "not a valid" in signature_problem("add_item(a,,)", "add_item")
    assert "must define the function 'add_item'" in signature_problem("other(a)", "add_item")


def errors_of(text):
    with pytest.raises(PlanError) as e:
        parse_plan(text)
    return e.value.errors


def test_plan_rejects_bad_signatures_and_stray_files():
    assert any("must define the function" in e for e in errors_of(plan(signature="other(cart)")))
    assert any("remove: cart/extra.py" in e for e in errors_of(
        plan(files=["cart/__init__.py", "cart/engine.py", "cart/extra.py"])))


def test_signature_needs_function_and_examples():
    assert any("needs 'function'" in e for e in errors_of(plan(function=None, examples=None, signature="f(x)")))


def test_bare_body_is_wrapped_in_the_signature():
    src, problem = assemble_module(step(), "    return [*cart, {'name': name}]\n")
    assert problem is None
    assert src == "\ndef add_item(cart, name, price, quantity=1):\n    return [*cart, {'name': name}]\n"


def test_leading_imports_are_hoisted_and_blank_lines_survive():
    src, _ = assemble_module(step(), "import copy\nout = copy.deepcopy(cart)\n\nreturn out")
    assert src.startswith("import copy\n\ndef add_item(")
    ns = {}
    exec(src, ns)
    assert ns["add_item"]([1], "a", 1) == [1]


def test_a_whole_function_written_by_the_model_is_accepted_as_is():
    whole = "def add_item(cart, name, price, quantity=1):\n    return cart\n"
    assert assemble_module(step(), whole) == (whole, None)


def test_empty_and_invalid_bodies_are_reported():
    assert "empty" in assemble_module(step(), "  \n")[1]
    assert "not valid Python" in assemble_module(step(), "return (")[1]


def test_init_file_reexports_the_function():
    assert init_source(step(), "cart/__init__.py") == "from .engine import add_item\n"


def test_nested_package_init_files():
    s = parse_plan(plan(function="a.b.c:add_item", files=["a/__init__.py", "a/b/__init__.py", "a/b/c.py"])).steps[0]
    assert init_source(s, "a/__init__.py") == "from .b.c import add_item\n"
    assert init_source(s, "a/b/__init__.py") == "from .c import add_item\n"


def test_assemble_files_builds_every_file_the_step_owns():
    files, problems = assemble_files(step(), {"cart/engine.py": "return cart"})
    assert problems == [] and set(files) == {"cart/engine.py", "cart/__init__.py"}
    assert files["cart/__init__.py"] == "from .engine import add_item\n"


def test_assemble_files_reports_a_bad_body():
    files, problems = assemble_files(step(), {"cart/engine.py": "return ("})
    assert files == {} and problems


def test_init_file_keeps_earlier_steps_reexports():
    files, _ = assemble_files(step(), {"cart/engine.py": "return cart"}, {"cart/__init__.py": "from .checks import check_args\n"})
    assert files["cart/__init__.py"] == "from .checks import check_args\nfrom .engine import add_item\n"


def test_init_line_is_not_duplicated_and_missing_newline_is_handled():
    again, _ = assemble_files(step(), {"cart/engine.py": "return cart"}, {"cart/__init__.py": "from .engine import add_item\n"})
    assert again["cart/__init__.py"] == "from .engine import add_item\n"
    nonl, _ = assemble_files(step(), {"cart/engine.py": "return cart"}, {"cart/__init__.py": "X = 1"})
    assert nonl["cart/__init__.py"] == "X = 1\nfrom .engine import add_item\n"
