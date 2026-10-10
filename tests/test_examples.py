from pathlib import Path

import pytest

from agent import Config, load_plan, parse_plan

EXAMPLES = sorted((Path(__file__).parent / "examples").glob("*.json"))


def test_there_are_examples():
    assert EXAMPLES


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_every_example_plan_is_valid(path):
    plan = load_plan(path)
    assert plan.test_command.split()[0] in Config().allowed_commands


GUIDE = Path(__file__).parent.parent / "docs" / "PLAN_GUIDE_FOR_AI.md"


def test_every_json_block_in_the_ai_guide_is_a_valid_plan():
    import re
    blocks = re.findall(r"```json\n(.*?)```", GUIDE.read_text(encoding="utf-8"), re.S)
    assert blocks
    for block in blocks:
        assert parse_plan(block).steps
