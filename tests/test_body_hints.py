import json

from agent import (
    Config,
    Runner,
    Workspace,
    build_impl_prompt,
    context_imports,
    parse_plan,
    step_context,
)
from tests.test_runner import BODY_OK, FakeGuard, FakeLLM, block, cfg  # noqa: F401

CHECKS = "def check_args(price, quantity):\n    return None\n"
MERGE = "def merge_quantity(cart, name, quantity):\n    return cart\n"
DESC = ("Create add_item(cart, name) in cart/engine.py. Use check_args from cart/checks.py and merge_quantity "
        "from cart/merge.py.\n1. Call check_args(price, quantity).\n2. Return merge_quantity(cart, name, 1).")


def make_plan(description=DESC, context=("cart/merge.py",), examples=None):
    step = {"title": "add", "description": description, "commit": "feat: add", "tests": ["tests/test_engine.py"],
            "files": ["cart/__init__.py", "cart/engine.py"], "context": list(context),
            "function": "cart.engine:add_item", "signature": "add_item(cart, name)",
            "examples": examples or [{"args": [[], "a"], "expect": []}]}
    return parse_plan(json.dumps({"title": "Cart", "test_command": "python -m pytest -q", "steps": [step]}))


def test_files_named_in_the_description_join_the_context_even_if_the_plan_forgot_them():
    step = make_plan().steps[0]
    assert step_context(step) == ("cart/merge.py", "cart/checks.py")
    assert context_imports(step, {"cart/merge.py": MERGE, "cart/checks.py": CHECKS}) == [
        "from cart.merge import merge_quantity", "from cart.checks import check_args"]


def test_the_step_own_files_and_tests_are_never_added_to_the_context():
    step = make_plan("Create add_item in cart/engine.py, tested by tests/test_engine.py.").steps[0]
    assert step_context(step) == ("cart/merge.py",)


def test_prompt_shows_and_imports_both_files():
    step = make_plan().steps[0]
    user = build_impl_prompt(make_plan(), step, {}, {"cart/merge.py": MERGE, "cart/checks.py": CHECKS},
                             6000, None, None)[1]["content"]
    assert "FILE: cart/checks.py" in user and "Already imported for you: merge_quantity, check_args" in user


def test_prompt_tells_the_model_to_follow_the_numbered_steps_only_when_there_are_some():
    step = make_plan().steps[0]
    user = build_impl_prompt(make_plan(), step, {}, {}, 6000, None, None)[1]["content"]
    assert "numbered steps" in user and "do not skip" in user.lower()
    plain = make_plan("Create add_item(cart, name) in cart/engine.py.")
    assert "numbered steps" not in build_impl_prompt(plain, plain.steps[0], {}, {}, 6000, None, None)[1]["content"]


def test_test_names_in_the_report_become_the_actual_calls():
    plan = make_plan(examples=[{"args": [[], "a"], "expect": []}, {"args": [[], "b"], "raises": "ValueError"}])
    out = ("test_example_1: IndexError: list index out of range\n"
           "test_example_2_raises_ValueError: Failed: DID NOT RAISE ValueError\n2 failed in 0.03s")
    user = build_impl_prompt(plan, plan.steps[0], {}, {}, 6000, out, None)[1]["content"]
    assert "add_item([], 'a'): IndexError: list index out of range" in user
    assert "add_item([], 'b') must raise ValueError, but it returned normally" in user
    assert "test_example_" not in user.split("What the tests reported:")[1]


def test_default_attempts_are_enough_for_fast_no_thinking_calls():
    assert Config().max_attempts >= 6


# --- runner ---------------------------------------------------------------------------

def run_with_util(tmp_path, cfg, llm):  # noqa: F811
    step = {"title": "core function", "description": "Create slugify(text) in slugify/core.py. Use clean from slugify/util.py.",
            "commit": "feat: add slugify", "tests": ["tests/test_core.py"],
            "files": ["slugify/__init__.py", "slugify/core.py"], "context": [],
            "function": "slugify.core:slugify", "signature": "slugify(text)",
            "examples": [{"args": ["  A "], "expect": "a"}]}
    plan = parse_plan(json.dumps({"title": "S", "test_command": "python -m pytest -q", "steps": [step]}))
    ws = Workspace(tmp_path / "proj", cfg.max_file_bytes)
    ws.write("slugify/util.py", "def clean(t):\n    return t.strip()\n")
    r = Runner(cfg, plan, ws, llm, guard=FakeGuard(), log=lambda m: None, sleep=lambda s: None,
               should_stop=lambda: False)
    return r.run(), ws


def test_a_file_only_named_in_the_description_is_imported_and_shown(tmp_path, cfg):  # noqa: F811
    llm = FakeLLM(block("slugify/core.py", "return clean(text).lower()"))
    result, ws = run_with_util(tmp_path, cfg, llm)
    assert result.outcome == "done" and "from slugify.util import clean" in ws.read("slugify/core.py")
    assert "FILE: slugify/util.py" in llm.calls[0][1]["content"]


def test_a_retry_that_hides_the_old_code_still_shows_the_related_files(tmp_path, cfg):  # noqa: F811
    bad = block("slugify/core.py", "return text")
    llm = FakeLLM(bad, bad, block("slugify/core.py", "return clean(text).lower()"))
    result, ws = run_with_util(tmp_path, cfg, llm)
    assert result.outcome == "done"
    third = llm.calls[2][1]["content"]
    assert "identical to your attempt 1" in third and "which failed" not in third
    assert "FILE: slugify/util.py" in third and "Already imported for you: clean" in third


def test_the_same_body_with_different_imports_still_counts_as_a_repeat(tmp_path, cfg):  # noqa: F811
    from tests.test_runner import make_body_runner
    lines = []
    llm = FakeLLM(block("slugify/core.py", "return text"), block("slugify/core.py", "import os\nreturn text"), BODY_OK)
    r, _ = make_body_runner(tmp_path, llm, cfg, log=lines.append)
    assert r.run().outcome == "done"
    assert any("same code as attempt 1" in ln for ln in lines)
