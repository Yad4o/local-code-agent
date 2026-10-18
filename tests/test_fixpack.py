import json

import pytest

from agent import (FixError, StepFailure, StepResult, apply_fix_pack, collect_failures, extract_json, parse_plan,
                   render_failure_report, write_failure_reports)
from tests.test_runner import examples_plan

BAD_DESC = "Create slugify(text) in slugify/core.py. Lowercase the text and join runs of other characters with one dash."
GOOD_DESC = ("Create slugify(text) in slugify/core.py.\n1. import re\n2. cleaned = re.sub(r'[^a-z0-9]+', '-', text.lower())\n"
             "3. return cleaned.strip('-')")


@pytest.fixture
def setup(tmp_path):
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(examples_plan(signature="slugify(text)", description=BAD_DESC), encoding="utf-8")
    plan = parse_plan(plan_file.read_text(encoding="utf-8"))
    ws = tmp_path / "ws"
    ws.mkdir()
    failure = StepFailure(plan.title, str(plan_file), plan.steps[0], "tests still failing after 6 attempts and 3 repair tries. Last line: 1 failed",
                          {"bodies": ["return text", "return text.lower()"], "output": "slugify('A b') returned 'a b', but it must return 'a-b'\n1 failed",
                           "notes": ["impl attempt 6: tests still failing"]})
    write_failure_reports(ws, "demo", [failure], ["next step"])
    return plan_file, ws, failure


def test_failures_are_told_apart_from_steps_that_only_wait_for_a_file():
    plan = parse_plan(examples_plan(signature="slugify(text)"))
    t = plan.steps[0].title
    failed = StepResult(t, "failed", 6, reason="tests still failing after 6 attempts", detail={"bodies": ["x"]})
    waiting = StepResult(t, "failed", 0, reason="the step description names a/b.py, but it is missing or empty in the workspace")
    assert [f.step.title for f in collect_failures(plan, "p.json", [failed])[0]] == [t]
    assert collect_failures(plan, "p.json", [waiting], {"a/b.py"}) == ([], [t])            # a failing step should have made a/b.py
    assert collect_failures(plan, "p.json", [StepResult(t, "done", 1)]) == ([], [])


def test_a_step_whose_missing_file_no_failing_step_makes_is_a_failure_of_its_own_never_waiting():
    # real bug: a wrong `context` path left 6 steps "waiting" for nothing and the build still said DONE
    plan = parse_plan(examples_plan(signature="slugify(text)"))
    t = plan.steps[0].title
    orphan = StepResult(t, "failed", 0, reason="the step description names ui_button, but it is missing or empty in the workspace")
    failures, blocked = collect_failures(plan, "p.json", [orphan])
    assert [f.step.title for f in failures] == [t] and blocked == []


def test_waiting_follows_a_chain_and_works_across_plans():
    plan = parse_plan(examples_plan(signature="slugify(text)"))
    t, files = plan.steps[0].title, set()
    first = StepResult(t, "failed", 6, reason="tests still failing after 6 attempts")
    assert len(collect_failures(plan, "p.json", [first], files)[0]) == 1 and plan.steps[0].files[1] in files
    later = StepResult(t, "failed", 0, reason=f"the step description names {plan.steps[0].files[1]}, but it is missing or empty")
    assert collect_failures(plan, "p2.json", [later], files) == ([], [t])


def test_the_report_has_everything_an_ai_needs_and_the_format_to_answer_in(setup):
    plan_file, ws, failure = setup
    text = (ws / ".codeagent" / "FAILURES.md").read_text(encoding="utf-8")
    for needle in (failure.id, str(plan_file), "return text.lower()", "must return 'a-b'", "impl attempt 6",
                   '"signature": "slugify(text)"', "Never reassign a parameter", '"fixes"', "codeagent fix fix.json",
                   "next step", "Reply with the JSON object and nothing else"):
        assert needle in text, needle
    assert text.startswith("# Help the local code agent: 1 failing step in demo")


def test_the_registry_maps_the_id_to_the_plan_and_step_and_is_removed_when_nothing_failed(setup):
    plan_file, ws, failure = setup
    reg = json.loads((ws / ".codeagent" / "failures.json").read_text())
    assert reg[failure.id]["plan_file"] == str(plan_file) and reg[failure.id]["step_title"] == failure.step.title
    assert write_failure_reports(ws, "demo", [], []) is None
    assert not (ws / ".codeagent" / "FAILURES.md").exists() and not (ws / ".codeagent" / "failures.json").exists()


@pytest.mark.parametrize("reply", [
    '{"fixes": []}',
    'Sure! Here is the fix:\n```json\n{"fixes": []}\n```\nHope it helps.',
    'The answer: {"fixes": []} done',
])
def test_the_json_is_found_in_an_ai_reply(reply):
    assert extract_json(reply) == {"fixes": []}


def test_a_reply_without_json_is_rejected_clearly():
    with pytest.raises(FixError, match="not JSON"):
        extract_json("I could not fix it, sorry.")


def fix(failure_id, **kw):
    return json.dumps({"fixes": [{"failure": failure_id, **kw}]})


def test_a_rewritten_description_goes_into_the_plan_with_a_backup(setup):
    plan_file, ws, failure = setup
    msgs = apply_fix_pack(ws, fix(failure.id, plan_step={"description": GOOD_DESC}), now=lambda: "T1")
    assert json.loads(plan_file.read_text())["steps"][0]["description"] == GOOD_DESC
    assert json.loads((plan_file.parent / "plan.json.bak-T1").read_text())["steps"][0]["description"] == BAD_DESC
    assert any("rewrote description" in m for m in msgs) and any("saved" in m and "bak-T1" in m for m in msgs)


def test_the_fixed_plan_is_a_valid_plan_and_a_bad_description_gets_a_warning(setup):
    plan_file, ws, failure = setup
    msgs = apply_fix_pack(ws, fix(failure.id, plan_step={"description": "Create slugify(text) in slugify/core.py.\n1. best = None and other = None\n2. return best"}))
    assert any(m.startswith("warning:") and "joined with 'and'" in m for m in msgs)
    parse_plan(plan_file.read_text())


def test_a_fix_that_would_break_the_plan_changes_nothing(setup):
    plan_file, ws, failure = setup
    before = plan_file.read_text()
    with pytest.raises(FixError, match="would be invalid"):
        apply_fix_pack(ws, fix(failure.id, plan_step={"examples": [{"args": [1]}]}))        # no expect/raises
    assert plan_file.read_text() == before and not list(plan_file.parent.glob("*.bak*"))


@pytest.mark.parametrize("bad,match", [
    ({"failure": "nope", "lessons": ["x"]}, "unknown failure id"),
    ({"lessons": ["x"]}, "say which step"),
    ({"failure": "ID", "extra": 1}, "unknown key"),
    ({"failure": "ID", "plan_step": {"title": "Another"}}, "do not rename"),
    ({"failure": "ID", "plan_step": {"function": "a.b:c"}}, "may only contain"),
    ({"failure": "ID", "body": "return ("}, "not valid Python"),
    ({"failure": "ID", "lessons": ["x" * 300]}, "longer than 240"),
    ({"failure": "ID", "lessons": ["a"] * 6}, "up to 5"),
    ({"failure": "ID", "lessons": ["a"], "scope": "universe"}, "scope"),
    ({"failure": "ID"}, "changes nothing"),
])
def test_every_kind_of_wrong_fix_is_refused_with_a_reason(setup, bad, match):
    plan_file, ws, failure = setup
    if bad.get("failure") == "ID":
        bad = {**bad, "failure": failure.id}
    with pytest.raises(FixError, match=match):
        apply_fix_pack(ws, json.dumps({"fixes": [bad]}))


def test_one_wrong_fix_in_a_pack_means_none_of_them_is_applied(setup, tmp_path, monkeypatch):
    plan_file, ws, failure = setup
    monkeypatch.setenv("CODEAGENT_LESSONS", str(tmp_path / "g.json"))
    before = plan_file.read_text()
    pack = {"fixes": [{"failure": failure.id, "plan_step": {"description": GOOD_DESC}, "lessons": ["a rule"]},
                      {"failure": failure.id, "body": "return ("}]}
    with pytest.raises(FixError):
        apply_fix_pack(ws, json.dumps(pack))
    assert plan_file.read_text() == before and not (ws / ".codeagent" / "lessons.json").exists()


def test_a_body_is_saved_for_the_next_run_to_try_first(setup):
    plan_file, ws, failure = setup
    apply_fix_pack(ws, fix(failure.id, body="    import re\n    return re.sub(r'x', 'y', text)"))
    saved = ws / ".codeagent" / "overrides" / f"{'slugify-lowercases-and-joins-with-dashes'}.py"
    files = list((ws / ".codeagent" / "overrides").glob("*.py"))
    assert len(files) == 1 and files[0].read_text() == "import re\nreturn re.sub(r'x', 'y', text)\n"


def test_lessons_go_to_the_project_or_to_every_project_without_duplicates(setup, tmp_path, monkeypatch):
    plan_file, ws, failure = setup
    monkeypatch.setenv("CODEAGENT_LESSONS", str(tmp_path / "g.json"))
    apply_fix_pack(ws, fix(failure.id, lessons=["Rule A.", "Rule B."]))
    msgs = apply_fix_pack(ws, fix(failure.id, lessons=["Rule B.", "Rule C."], scope="global"))
    assert json.loads((ws / ".codeagent" / "lessons.json").read_text())["lessons"] == ["Rule A.", "Rule B."]
    assert json.loads((tmp_path / "g.json").read_text())["lessons"] == ["Rule B.", "Rule C."]
    assert any("global" in m for m in msgs)
    apply_fix_pack(ws, fix(failure.id, lessons=["Rule A.", "Rule D."]))
    assert json.loads((ws / ".codeagent" / "lessons.json").read_text())["lessons"] == ["Rule A.", "Rule B.", "Rule D."]


def test_a_single_fix_object_without_the_fixes_list_is_accepted(setup):
    plan_file, ws, failure = setup
    apply_fix_pack(ws, json.dumps({"failure": failure.id, "plan_step": {"description": GOOD_DESC}}))
    assert json.loads(plan_file.read_text())["steps"][0]["description"] == GOOD_DESC


def test_the_step_can_be_found_by_its_title_when_the_id_is_left_out(setup):
    plan_file, ws, failure = setup
    apply_fix_pack(ws, json.dumps({"fixes": [{"plan_step": {"title": failure.step.title, "description": GOOD_DESC}}]}))
    assert json.loads(plan_file.read_text())["steps"][0]["description"] == GOOD_DESC
