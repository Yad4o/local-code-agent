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


def test_every_json_block_in_the_ai_guide_is_valid():
    import json
    import re
    blocks = re.findall(r"```json\n(.*?)```", GUIDE.read_text(encoding="utf-8"), re.S)
    kinds = {"plan": 0, "project": 0, "groups": 0, "fix pack": 0}
    for block in blocks:
        data = json.loads(block)
        if "steps" in data:
            assert parse_plan(block).steps
            kinds["plan"] += 1
        elif "plans" in data:
            assert set(data) <= {"name", "description", "plans", "groups", "workspace"} and data["name"] and data["plans"]
            kinds["project"] += 1
        elif "modules" in data:
            assert data["package"].isidentifier() and all(isinstance(v, list) and v for v in data["modules"].values())
            kinds["groups"] += 1
        elif "fixes" in data:
            assert set(data["fixes"][0]) <= {"failure", "plan_step", "body", "lessons", "scope"}
            kinds["fix pack"] += 1
        else:
            raise AssertionError(f"a JSON block of an unknown kind: {block[:60]}")
    assert all(kinds.values()), kinds
