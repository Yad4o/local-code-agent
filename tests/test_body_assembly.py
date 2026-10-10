import ast

from agent import (
    Config,
    assemble_files,
    assemble_module,
    build_impl_prompt,
    context_imports,
    parse_plan,
)
from tests.test_runner import (  # noqa: F401
    FakeLLM,
    block,
    cfg,
    examples_plan,
    make_body_runner,
)

CHECKS = "def check_args(price, quantity):\n    return None\n"
MERGE = "def merge_quantity(cart, name, quantity):\n    return cart\n"


def make_step(**changes):
    d = dict(title="add", description="Create add_item(cart, name) in cart/engine.py. Use check_args from "
             "cart/checks.py and merge_quantity from cart/merge.py.", commit="feat: add",
             tests=["tests/test_engine.py"], files=["cart/__init__.py", "cart/engine.py"],
             context=["cart/checks.py", "cart/merge.py", "cart/other.py"], function="cart.engine:add_item",
             signature="add_item(cart, name)", examples=[{"args": [[], "a"], "expect": []}])
    d.update(changes)
    import json
    return parse_plan(json.dumps({"title": "Cart", "test_command": "python -m pytest -q", "steps": [d]})).steps[0]


SOURCES = {"cart/checks.py": CHECKS, "cart/merge.py": MERGE, "cart/other.py": "def unrelated():\n    pass\n"}


def test_context_functions_named_in_the_description_are_imported_for_the_model():
    assert context_imports(make_step(), SOURCES) == [
        "from cart.checks import check_args", "from cart.merge import merge_quantity"]


def test_unmentioned_functions_missing_files_and_the_step_itself_are_not_imported():
    step = make_step(description="Create add_item(cart, name). It does a simple thing.",
                     context=["cart/other.py", "cart/missing.py", "cart/engine.py"])
    assert context_imports(step, {**SOURCES, "cart/engine.py": "def add_item(cart, name):\n    pass\n"}) == []


def test_imports_go_on_top_and_are_not_duplicated_when_the_model_wrote_them_too():
    step = make_step()
    files, problems = assemble_files(step, {"cart/engine.py": "return merge_quantity(cart, name, 1)"}, {}, SOURCES)
    assert problems == [] and files["cart/engine.py"].startswith(
        "from cart.checks import check_args\nfrom cart.merge import merge_quantity\n")
    body = "from cart.merge import merge_quantity\nreturn merge_quantity(cart, name, 1)"
    files, _ = assemble_files(step, {"cart/engine.py": body}, {}, SOURCES)
    assert files["cart/engine.py"].count("from cart.merge import merge_quantity") == 1
    ns = {}
    exec("def check_args(p, q): pass\ndef merge_quantity(c, n, q): return 'ok'\n", ns)
    src = files["cart/engine.py"].replace("from cart.checks import check_args\n", "").replace(
        "from cart.merge import merge_quantity\n", "")
    exec(src, ns)
    assert ns["add_item"]([], "a") == "ok"


def test_a_def_written_by_the_model_gets_the_planned_signature_back():
    step = make_step(signature="add_item(cart, name, price, quantity=1)")
    src, problem = assemble_module(step, "def add_item(cart, name, price=1):\n    return cart\n")
    assert problem is None
    fn = ast.parse(src).body[0]
    assert ast.unparse(fn.args) == "cart, name, price, quantity=1"


def test_the_prompt_says_what_is_already_imported():
    plan = parse_plan(__import__("json").dumps({"title": "Cart", "test_command": "python -m pytest -q",
                                               "steps": [{"title": "add", "description": "Use check_args from cart/checks.py.",
                                                          "commit": "feat: add", "tests": ["tests/t.py"],
                                                          "files": ["cart/__init__.py", "cart/engine.py"],
                                                          "context": ["cart/checks.py"], "function": "cart.engine:add_item",
                                                          "signature": "add_item(cart)",
                                                          "examples": [{"args": [[]], "expect": []}]}]}))
    user = build_impl_prompt(plan, plan.steps[0], {}, {"cart/checks.py": CHECKS}, 6000, None, None)[1]["content"]
    assert "Already imported for you: check_args" in user


# --- thinking is off for body steps, and cooler when it is -----------------------------

def test_body_steps_do_not_think_by_default_and_run_cooler(tmp_path, cfg):  # noqa: F811
    from tests.test_runner import BODY_OK
    llm = FakeLLM(BODY_OK)
    make_body_runner(tmp_path, llm, cfg)[0].run()
    assert llm.kwargs[0]["think"] is False
    assert llm.kwargs[0]["temperature"] == Config().temperature_no_think


def test_think_body_true_keeps_thinking_for_the_first_attempt(tmp_path, cfg):  # noqa: F811
    from agent import NoAnswer
    from tests.test_runner import BODY_OK
    cfg.think_body = True
    llm = FakeLLM(NoAnswer("stuck repeating itself"), BODY_OK)
    make_body_runner(tmp_path, llm, cfg)[0].run()
    assert "think" not in llm.kwargs[0] and llm.kwargs[1]["think"] is False


def test_a_syntax_error_in_the_body_quotes_the_bad_line_in_the_feedback(tmp_path, cfg):  # noqa: F811
    from tests.test_runner import BODY_OK, block
    bad = block("slugify/core.py", "best = None and other = None\nreturn text")
    llm = FakeLLM(bad, BODY_OK)
    make_body_runner(tmp_path, llm, cfg)[0].run()
    second_prompt = str(llm.calls[1])
    assert "best = None and other = None" in second_prompt


def test_a_reply_that_is_bare_code_without_a_fence_is_used_instead_of_wasting_an_attempt(tmp_path, cfg):  # noqa: F811
    bare = "import re\ndef slugify(text):\n    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')\n"
    llm = FakeLLM(bare)
    report = make_body_runner(tmp_path, llm, cfg)[0].run()
    assert report.results[0].status == "done" and len(llm.kwargs) == 1


def test_prose_without_a_fence_is_still_rejected(tmp_path, cfg):  # noqa: F811
    from tests.test_runner import BODY_OK
    llm = FakeLLM("Sure! Here is how I would do it, first lowercase the text.", BODY_OK)
    report = make_body_runner(tmp_path, llm, cfg)[0].run()
    assert report.results[0].status == "done" and len(llm.kwargs) == 2


def test_a_body_that_keeps_failing_is_retried_with_thinking_from_attempt_think_body_after(tmp_path, cfg):  # noqa: F811
    from tests.test_runner import BODY_BAD, BODY_OK
    cfg.think_body_after = 3
    bad2 = BODY_BAD.replace("return text", "return text.lower()")
    llm = FakeLLM(BODY_BAD, bad2, BODY_OK)
    make_body_runner(tmp_path, llm, cfg)[0].run()
    assert [k.get("think") for k in llm.kwargs[:3]] == [False, False, None]


def test_think_body_after_zero_never_turns_thinking_back_on(tmp_path, cfg):  # noqa: F811
    from tests.test_runner import BODY_BAD, BODY_OK
    cfg.think_body_after = 0
    llm = FakeLLM(BODY_BAD, BODY_BAD.replace("return text", "return text.lower()"), BODY_OK)
    make_body_runner(tmp_path, llm, cfg)[0].run()
    assert [k.get("think") for k in llm.kwargs[:3]] == [False, False, False]


def test_a_body_that_relies_on_a_context_function_passes_without_writing_the_import(tmp_path, cfg):  # noqa: F811
    import json

    from agent import Runner, Workspace
    from tests.test_runner import FakeGuard
    step = {"title": "core function", "description": "Create slugify(text) in slugify/core.py. Use clean from slugify/util.py.",
            "commit": "feat: add slugify", "tests": ["tests/test_core.py"],
            "files": ["slugify/__init__.py", "slugify/core.py"], "context": ["slugify/util.py"],
            "function": "slugify.core:slugify", "signature": "slugify(text)",
            "examples": [{"args": ["  A "], "expect": "a"}]}
    plan = parse_plan(json.dumps({"title": "S", "test_command": "python -m pytest -q", "steps": [step]}))
    ws = Workspace(tmp_path / "proj", cfg.max_file_bytes)
    ws.write("slugify/util.py", "def clean(t):\n    return t.strip()\n")
    llm = FakeLLM(block("slugify/core.py", "return clean(text).lower()"))
    r = Runner(cfg, plan, ws, llm, guard=FakeGuard(), log=lambda m: None, sleep=lambda s: None,
               should_stop=lambda: False)
    assert r.run().outcome == "done" and len(llm.calls) == 1
    assert "from slugify.util import clean" in ws.read("slugify/core.py")
