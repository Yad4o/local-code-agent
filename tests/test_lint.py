import re
from pathlib import Path

import pytest

from agent import lint_description, load_plan, parse_plan

ROOT = Path(__file__).parent


def test_two_assignments_joined_with_and_are_flagged():
    # copied verbatim by the model into invalid Python, 6 attempts in a row
    assert lint_description("2. best_name = None and best_avg = None\n")
    assert not lint_description("2. best_name = None\n3. best_avg = None\n")


def test_comparisons_with_and_are_not_flagged():
    assert not lint_description('1. if score >= 90 and score <= 100: return "A"\n2. if a == b and c != d: return 1\n')


def test_a_loop_and_its_body_on_one_line_is_flagged():
    assert lint_description("2. For each ch in text.lower(): if ch.isalnum(): cleaned = cleaned + ch\n")
    assert not lint_description("2. For each ch in text.lower():\n   - if ch.isalnum(): cleaned = cleaned + ch\n")


def test_semicolon_joined_statements_are_flagged():
    assert lint_description('   - if x: a = 1; b = 2\n')


def test_an_english_condition_is_flagged_and_the_code_form_is_not():
    assert lint_description("2. If len(text) <= limit, return text.\n")
    assert not lint_description("2. if len(text) <= limit: return text\n")


def test_long_recipes_and_two_loops_are_flagged_but_a_six_line_flat_chain_is_not():
    seven = "".join(f"{i}. x = {i}\n" for i in range(1, 8))
    assert lint_description(seven)
    six = "".join(f"{i}. x = {i}\n" for i in range(1, 7))
    assert not lint_description(six)
    assert lint_description("1. For each a in b:\n   - x = a\n2. For each c in d:\n   - y = c\n")


EXAMPLES = sorted((ROOT / "examples").glob("*.json"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_every_example_plan_is_free_of_lint_warnings(path):
    for step in load_plan(path).steps:
        assert lint_description(step.description) == [], step.title


def test_every_plan_in_the_ai_guide_is_free_of_lint_warnings():
    md = (ROOT.parent / "docs" / "PLAN_GUIDE_FOR_AI.md").read_text(encoding="utf-8")
    for block in re.findall(r"```json\n(.*?)```", md, re.S):
        for step in parse_plan(block).steps:
            assert lint_description(step.description) == [], step.title
