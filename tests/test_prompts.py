from codeagent.plan import Plan, Step
from codeagent.prompts import SYSTEM, build_impl_prompt, build_tests_prompt

STEP = Step(
    title="core function",
    description="Create slugify(text) in slugify/core.py. Example: 'Hello World' -> 'hello-world'.",
    commit="feat: add slugify core function",
    tests=("tests/test_core.py",),
    files=("slugify/__init__.py", "slugify/core.py"),
    context=("README.md",),
)
PLAN = Plan("Slugify utility", "python -m pytest -q", None, (STEP,))


def text(msgs):
    return "\n".join(m["content"] for m in msgs)


def test_messages_are_system_then_user():
    for msgs in (build_tests_prompt(PLAN, STEP, {}, 6000, None),
                 build_impl_prompt(PLAN, STEP, {"tests/test_core.py": "def test_x(): pass\n"}, {}, 6000, None, None)):
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert msgs[0]["content"] == SYSTEM


def test_system_prompt_teaches_the_exact_output_format():
    assert "FILE:" in SYSTEM and "```" in SYSTEM
    assert "COMPLETE" in SYSTEM or "complete" in SYSTEM


def test_tests_prompt_has_task_description_and_target_files():
    t = text(build_tests_prompt(PLAN, STEP, {}, 6000, None))
    assert "Slugify utility" in t
    assert "Hello World" in t and "hello-world" in t
    assert "tests/test_core.py" in t
    assert "slugify/core.py" in t               # tells the model where the code will live
    assert "FAIL" in t.upper()                    # tests must fail first (TDD)


def test_tests_prompt_includes_feedback_when_given():
    t = text(build_tests_prompt(PLAN, STEP, {}, 6000, "your tests passed without any implementation"))
    assert "your tests passed without any implementation" in t


def test_tests_prompt_shows_existing_files_and_context():
    existing = {"slugify/core.py": "def slugify(t):\n    return t\n", "README.md": "# Readme text"}
    t = text(build_tests_prompt(PLAN, STEP, existing, 6000, None))
    assert "def slugify(t):" in t
    assert "# Readme text" in t


def test_impl_prompt_contains_tests_and_forbids_changing_them():
    tests = {"tests/test_core.py": "from slugify.core import slugify\ndef test_a():\n    assert slugify('A B') == 'a-b'\n"}
    t = text(build_impl_prompt(PLAN, STEP, tests, {}, 6000, None, None))
    assert "assert slugify('A B') == 'a-b'" in t
    assert "do not" in t.lower() and "tests" in t.lower()
    assert "slugify/core.py" in t and "slugify/__init__.py" in t


def test_impl_prompt_includes_failure_output_and_current_code():
    existing = {"slugify/core.py": "def slugify(t):\n    return 'WRONG'\n"}
    out = "E   AssertionError: assert 'WRONG' == 'a-b'"
    t = text(build_impl_prompt(PLAN, STEP, {"tests/test_core.py": "x\n"}, existing, 6000, out, None))
    assert "return 'WRONG'" in t
    assert "AssertionError: assert 'WRONG' == 'a-b'" in t


def test_impl_prompt_includes_format_feedback():
    t = text(build_impl_prompt(PLAN, STEP, {"tests/test_core.py": "x\n"}, {}, 6000, None,
                           "reply had no FILE blocks"))
    assert "reply had no FILE blocks" in t


def test_large_files_are_truncated_to_the_budget():
    existing = {"slugify/core.py": "x = 1\n" * 5000}
    t = text(build_tests_prompt(PLAN, STEP, existing, 1000, None))
    assert len(t) < 4000
    assert "truncated" in t


def test_budget_is_shared_between_files():
    big = "line\n" * 3000
    existing = {"slugify/core.py": big, "README.md": big}
    t = text(build_tests_prompt(PLAN, STEP, existing, 2000, None))
    assert t.count("line") < 800                  # both truncated, neither hogs the budget
    assert "slugify/core.py" in t and "README.md" in t


def test_prompts_never_mention_step_numbers_or_internal_ids():
    t = text(build_tests_prompt(PLAN, STEP, {}, 6000, None)).lower()
    assert "task 1" not in t and "step 1" not in t
