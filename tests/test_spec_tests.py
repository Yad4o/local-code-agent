import ast

from codeagent.plan import parse_plan
from codeagent.spec_tests import defined_names, render_tests

PLAN = '''
title = "Cart"
test_command = "python -m pytest -q"

[[step]]
title = "add"
description = "add_item"
commit = "feat: add"
tests = ["tests/test_add.py"]
files = ["cart/__init__.py", "cart/engine.py"]
function = "cart.engine:add_item"

[[step.examples]]
args = [[], "apple", 1.5]
expect = [{ name = "apple", price = 1.5, quantity = 1 }]

[[step.examples]]
args = [[], "apple", -1.0]
raises = "ValueError"

[[step.examples]]
args = [[], "apple", 1.5]
kwargs = { quantity = 0 }
raises = "ValueError"
'''


def step():
    return parse_plan(PLAN).steps[0]


def test_rendered_tests_are_valid_python_with_unique_names():
    src = render_tests(step())
    tree = ast.parse(src)
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert len(names) == 3 == len(set(names))


def test_rendered_tests_call_the_function_exactly_as_the_plan_says():
    src = render_tests(step())
    assert "from cart.engine import add_item" in src and "import pytest" in src
    assert "assert add_item([], 'apple', 1.5) == [{'name': 'apple', 'price': 1.5, 'quantity': 1}]" in src
    assert "with pytest.raises(ValueError):" in src
    assert "add_item([], 'apple', 1.5, quantity=0)" in src


def test_no_pytest_import_when_nothing_raises():
    plan = parse_plan(PLAN.split("[[step.examples]]")[0] + "[[step.examples]]\nargs = [[], 'a', 1.0]\nexpect = []\n")
    assert "import pytest" not in render_tests(plan.steps[0])


def test_rendered_tests_actually_run(tmp_path):
    import subprocess, sys
    (tmp_path / "cart").mkdir()
    (tmp_path / "cart" / "__init__.py").write_text("")
    (tmp_path / "cart" / "engine.py").write_text(
        "def add_item(cart, name, price, quantity=1):\n"
        "    if price < 0 or quantity <= 0:\n        raise ValueError('bad')\n"
        "    return [*cart, {'name': name, 'price': price, 'quantity': quantity}]\n")
    (tmp_path / "test_add.py").write_text(render_tests(step()))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "test_add.py"], cwd=tmp_path,
                       capture_output=True, text=True)
    assert "3 passed" in r.stdout, r.stdout


def test_defined_names():
    src = "import os\nfrom .a import b as c\nX = 1\ndef f(): pass\nclass K: pass\n"
    assert defined_names(src) == {"os", "c", "X", "f", "K"}
    assert defined_names("def broken(:") == set()


def test_returns_a_value():
    from codeagent.spec_tests import returns_a_value
    assert returns_a_value("def f(x):\n    return x + 1\n", "f") is True
    assert returns_a_value("def f(x):\n    x.append(1)\n", "f") is False
    assert returns_a_value("def f(x):\n    return\n", "f") is False
    assert returns_a_value("def f(x):\n    return None\n", "f") is False
    assert returns_a_value("def f(x):\n    yield x\n", "f") is True
    assert returns_a_value("def f(x):\n    def g():\n        return 1\n    x.append(g)\n", "f") is False
    assert returns_a_value("def f(x):\n    if x:\n        return 1\n", "f") is True
    assert returns_a_value("def other(): return 1\n", "f") is None
    assert returns_a_value("def f(:\n", "f") is None
