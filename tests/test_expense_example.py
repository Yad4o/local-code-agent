"""The three Expense Manager plans (38 steps) run one after another in the same workspace, as the guide advises,
with reference bodies standing in for the model. Includes non-ASCII output (the rupee sign) and cross-plan context files."""
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from agent import Config, Runner, Workspace, load_plan, missing_recipe_lines
from tests.expense_refs import REF
from tests.test_runner import FakeGuard, FakeLLM, block

EXAMPLES = Path(__file__).parent / "examples"
PLANS = ["expense_core", "expense_reports", "expense_editing"]


def _body(function: str) -> str:
    return textwrap.dedent("\n".join(REF[function].strip("\n").splitlines()[1:]))


@pytest.mark.parametrize("name", PLANS)
def test_a_good_body_never_triggers_the_missing_lines_hint(name):
    for step in load_plan(EXAMPLES / f"{name}.json").steps:
        assert missing_recipe_lines(step.description, _body(step.function)) == [], step.title


def test_the_three_plans_build_in_one_workspace_and_every_test_passes(tmp_path):
    cfg = Config()
    ws = Workspace(tmp_path / "proj", cfg.max_file_bytes)
    total = 0
    for name in PLANS:
        plan = load_plan(EXAMPLES / f"{name}.json")
        replies = [block(step.files[1], _body(step.function)) for step in plan.steps]
        runner = Runner(cfg, plan, ws, FakeLLM(*replies), guard=FakeGuard(), log=lambda m: None,
                        sleep=lambda s: None, should_stop=lambda: False)
        report = runner.run()
        assert [s.status for s in report.results] == ["done"] * len(plan.steps), name
        total += len(plan.steps)
    assert total == 38
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=tmp_path / "proj", capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-800:]
