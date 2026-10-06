import json

import pytest

from codeagent.plan import Plan, Step
from codeagent.state import RunState, WorkspaceBusy, WorkspaceLock

STEP = Step("core", "do the thing", "feat: core", ("tests/t.py",), ("pkg/c.py",))
PLAN = Plan("My plan", "pytest", None, (STEP,))


def test_new_state_has_nothing_done(tmp_path):
    assert RunState(tmp_path).is_done(PLAN, STEP) is False


def test_mark_done_persists_across_instances(tmp_path):
    RunState(tmp_path).mark_done(PLAN, STEP, "abc1234")
    assert RunState(tmp_path).is_done(PLAN, STEP) is True


def test_changed_step_content_is_not_considered_done(tmp_path):
    RunState(tmp_path).mark_done(PLAN, STEP, "abc")
    edited = Step("core", "do a DIFFERENT thing", "feat: core", ("tests/t.py",), ("pkg/c.py",))
    assert RunState(tmp_path).is_done(PLAN, edited) is False


def test_plans_are_tracked_independently(tmp_path):
    RunState(tmp_path).mark_done(PLAN, STEP, "abc")
    other = Plan("Other plan", "pytest", None, (STEP,))
    assert RunState(tmp_path).is_done(other, STEP) is False


def test_state_file_lives_in_dot_codeagent_and_is_valid_json(tmp_path):
    RunState(tmp_path).mark_done(PLAN, STEP, "abc")
    data = json.loads((tmp_path / ".codeagent" / "state.json").read_text())
    assert data["plans"]["My plan"]["core"]["commit"] == "abc"


def test_corrupted_state_is_survivable_and_backed_up(tmp_path):
    d = tmp_path / ".codeagent"
    d.mkdir()
    (d / "state.json").write_text("{ not json")
    s = RunState(tmp_path)
    assert s.is_done(PLAN, STEP) is False
    s.mark_done(PLAN, STEP, "abc")                         # still usable afterwards
    assert RunState(tmp_path).is_done(PLAN, STEP) is True
    assert (d / "state.json.corrupt").read_text() == "{ not json"


def test_saving_leaves_no_temp_files(tmp_path):
    RunState(tmp_path).mark_done(PLAN, STEP, "abc")
    assert sorted(p.name for p in (tmp_path / ".codeagent").iterdir()) == ["state.json"]


def test_lock_prevents_a_second_holder(tmp_path):
    with WorkspaceLock(tmp_path):
        with pytest.raises(WorkspaceBusy):
            with WorkspaceLock(tmp_path):
                pass


def test_lock_is_released_afterwards(tmp_path):
    with WorkspaceLock(tmp_path):
        pass
    with WorkspaceLock(tmp_path):
        pass


def test_lock_is_released_after_an_exception(tmp_path):
    with pytest.raises(RuntimeError):
        with WorkspaceLock(tmp_path):
            raise RuntimeError("boom")
    with WorkspaceLock(tmp_path):
        pass
