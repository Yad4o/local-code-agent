from pathlib import Path

import pytest

from agent import Config, load_plan

EXAMPLES = sorted((Path(__file__).parent / "examples").glob("*.json"))


def test_there_are_examples():
    assert EXAMPLES


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_every_example_plan_is_valid(path):
    plan = load_plan(path)
    assert plan.test_command.split()[0] in Config().allowed_commands
