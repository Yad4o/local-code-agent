from pathlib import Path

import pytest

from codeagent.config import Config
from codeagent.plan import load_plan

EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples").glob("*.toml"))


def test_there_are_examples():
    assert EXAMPLES


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_every_example_plan_is_valid(path):
    plan = load_plan(path)
    assert plan.test_command.split()[0] in Config().allowed_commands
