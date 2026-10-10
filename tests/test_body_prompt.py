from agent import (
    SYSTEM,
    SYSTEM_BODY,
    build_impl_prompt,
    condense_test_output,
    function_body,
    parse_plan,
)
from tests.test_runner import (  # noqa: F401
    FakeLLM,
    block,
    cfg,
    examples_plan,
    make_body_runner,
)

PLAN = parse_plan(examples_plan(signature="slugify(text)"))
STEP = PLAN.steps[0]
OLD = "def slugify(text):\n    if len(text) <= 0:\n        return ''\n    return text\n"
INIT = "from .core import slugify\n"
EXISTING = {"slugify/core.py": OLD, "slugify/__init__.py": INIT}
TESTS = {"tests/test_core.py": "def test_example_1():\n    assert 1\n"}


def prompt(existing=None, out=None, fb=None):
    return build_impl_prompt(PLAN, STEP, TESTS, EXISTING if existing is None else existing, 6000, out, fb)


def test_body_mode_has_its_own_system_prompt_that_does_not_ask_for_a_complete_file():
    msgs = prompt()
    assert msgs[0]["content"] == SYSTEM_BODY != SYSTEM
    assert "COMPLETE" not in SYSTEM_BODY and "body" in SYSTEM_BODY.lower() and "FILE:" in SYSTEM_BODY


def test_examples_are_listed_instead_of_the_pytest_file():
    user = prompt()[1]["content"]
    assert "slugify('Hello, World!') -> 'hello-world'" in user
    assert "assert result == expected" not in user and "def test_example_1" not in user


def test_the_package_init_and_the_whole_def_file_are_not_shown_only_the_failed_body():
    user = prompt()[1]["content"]
    assert "from .core import slugify" not in user
    assert "def slugify(text):\n    if len" not in user
    assert "which failed" in user and "if len(text) <= 0:" in user


def test_a_placeholder_body_is_not_shown_as_a_failed_attempt():
    user = prompt({"slugify/core.py": "def slugify(text):\n    pass\n"})[1]["content"]
    assert "which failed" not in user
    assert "Write ONLY its body" in user


def test_the_reply_instruction_is_last_and_names_the_file():
    user = prompt(out="1 failed", fb="bad")[1]["content"]
    assert user.rstrip().endswith("```") and user.rindex("FILE: slugify/core.py") > user.index("Problem with your previous")


def test_function_body_extracts_the_lines_inside_the_function():
    assert function_body("import re\n\ndef f(a):\n    x = 1\n    return x\n", "f") == "x = 1\nreturn x"
    assert function_body("def g(): pass\n", "f") is None and function_body("def f(:\n", "f") is None


RAW = """...[output truncated, showing last 3000 chars]
rror: merge_quantity([1], 'apple', 2) returned [1], but it must return [3]
E       assert [1] == [3]
E         
E         At index 0 diff: 1 != 3
E         Use -v to get more diff

tests/test_merge.py:7: AssertionError
________________________________ test_example_2 ________________________________

    def test_example_2():
>       assert result == expected, "long call text" + ' returned ' + repr(result)
E       AssertionError: merge_quantity([1, 2], 'pear', 1) returned [2], but it must return [1, 3]
E       assert [2] == [1, 3]
E         Right contains one more item: 3
E         Use -v to get more diff

tests/test_merge.py:13: AssertionError
________________________________ test_example_3 ________________________________
E       NameError: name 'zzz' is not defined
=========================== short test summary info ============================
FAILED tests/test_merge.py::test_example_2 - AssertionError: merge_quantity(...
3 failed, 4 passed in 0.03s
"""


def test_condensed_output_keeps_one_clear_line_per_failure_and_the_summary():
    out = condense_test_output(RAW)
    assert "test_example_2: AssertionError: merge_quantity([1, 2], 'pear', 1) returned [2], but it must return [1, 3]" in out
    assert "test_example_3: NameError: name 'zzz' is not defined" in out
    assert out.rstrip().endswith("3 failed, 4 passed in 0.03s")
    for noise in ("Use -v", "At index", "output truncated", "long call text", "assert [2] == [1, 3]"):
        assert noise not in out


def test_unrecognised_output_falls_back_to_the_tail_cut_at_a_line_boundary():
    raw = "\n".join(f"line number {i}" for i in range(500))
    out = condense_test_output(raw, limit=200)
    assert len(out) <= 200 and out.endswith("line number 499") and out.splitlines()[0].startswith("line number")


def test_many_failures_are_capped():
    raw = "\n".join(f"{'_' * 5} test_{i} {'_' * 5}\nE       AssertionError: bad {i}" for i in range(30))
    out = condense_test_output(raw)
    assert "bad 0" in out and "more" in out and "bad 29" not in out


def test_a_leftover_wrong_module_is_reset_so_the_first_prompt_shows_an_honest_failure(tmp_path, cfg):  # noqa: F811
    from tests.test_runner import BODY_OK
    llm = FakeLLM(BODY_OK)
    r, ws = make_body_runner(tmp_path, llm, cfg)
    ws.write("slugify/core.py", OLD)                        # what a killed earlier run leaves behind
    ws.write("slugify/__init__.py", INIT)
    assert r.run().outcome == "done"
    first = llm.calls[0][1]["content"]
    assert "if len(text) <= 0" not in first and "which failed" not in first
    assert "returned None" in first                         # the tests ran against the clean `pass` body
