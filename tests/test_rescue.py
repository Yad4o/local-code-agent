import json

import pytest

from agent import (SYSTEM_BODY, SYSTEM_RESCUE, Config, Runner, Workspace, compile_recipe, load_lessons, override_path,
                   parse_plan)
from tests.test_runner import BODY_BAD, BODY_OK, FakeGuard, block, examples_plan

CODE_RECIPE = ("Create slugify(text) in slugify/core.py.\n1. import re\n2. cleaned = re.sub(r'[^a-z0-9]+', '-', text.lower())\n"
               "3. return cleaned.strip('-')")
ENGLISH_RECIPE = "Create slugify(text) in slugify/core.py. Lowercase the text, turn every run of other characters into one dash, trim dashes."


class ScriptedLLM:
    """Answers by looking at the prompt: normal prompts get `normal`, repair prompts get `repair`."""
    def __init__(self, normal, repair=None):
        self.normal, self.repair, self.calls, self.kinds = normal, repair, [], []

    def chat(self, messages, **kw):
        self.calls.append(messages)
        kind = "repair" if messages[0]["content"] == SYSTEM_RESCUE else "normal"
        self.kinds.append(kind)
        reply = self.repair if kind == "repair" else self.normal
        return reply[min(len(self.kinds) - 1, len(reply) - 1)] if isinstance(reply, list) else reply


def runner(tmp_path, llm, description, **cfg_kw):
    plan = parse_plan(examples_plan(signature="slugify(text)", description=description))
    cfg = Config(step_cooldown_s=0, max_attempts=2, test_timeout=60, **cfg_kw)
    ws = Workspace(tmp_path / "proj", cfg.max_file_bytes)
    return Runner(cfg, plan, ws, llm, guard=FakeGuard(), log=lambda m: None, sleep=lambda s: None,
                  should_stop=lambda: False), ws


def test_when_the_attempts_run_out_the_recipe_is_tried_as_code_with_no_more_model_calls(tmp_path):
    llm = ScriptedLLM(BODY_BAD)
    r, ws = runner(tmp_path, llm, CODE_RECIPE)
    result = r.run().results[0]
    assert result.status == "done" and result.via == "recipe"
    assert llm.kinds == ["normal", "normal"]                           # only the two normal attempts
    assert "re.sub" in (ws.root / "slugify" / "core.py").read_text()


def test_an_english_recipe_gets_a_different_repair_prompt_with_the_draft_and_the_report(tmp_path):
    llm = ScriptedLLM(BODY_BAD, repair=BODY_OK)
    r, _ = runner(tmp_path, llm, ENGLISH_RECIPE)
    result = r.run().results[0]
    assert result.status == "done" and result.via == "repair prompt"
    assert llm.kinds == ["normal", "normal", "repair"]
    user = llm.calls[2][1]["content"]
    assert "does not pass the tests yet" in user and "return text" in user and "What the tests reported" in user
    assert llm.calls[2][0]["content"] != SYSTEM_BODY


def test_a_step_that_cannot_be_rescued_fails_with_everything_needed_to_ask_for_help(tmp_path):
    llm = ScriptedLLM(BODY_BAD, repair=BODY_BAD)
    r, ws = runner(tmp_path, llm, ENGLISH_RECIPE, rescue_attempts=2)
    result = r.run().results[0]
    assert result.status == "failed" and "2 repair tries" in result.reason
    assert llm.kinds.count("repair") == 2
    d = result.detail
    assert d["bodies"] and "return text" in d["bodies"][-1] and "slugify(" in d["output"] and d["notes"]
    assert not (ws.root / "slugify" / "core.py").exists() or "return text" not in (ws.root / "slugify" / "core.py").read_text()


def test_rescue_can_be_switched_off(tmp_path):
    llm = ScriptedLLM(BODY_BAD)
    r, _ = runner(tmp_path, llm, CODE_RECIPE, rescue=False)
    assert r.run().results[0].status == "failed" and llm.kinds == ["normal", "normal"]


def test_a_body_supplied_through_fix_is_used_first_and_needs_no_model(tmp_path):
    r, ws = runner(tmp_path, ScriptedLLM(BODY_BAD), ENGLISH_RECIPE)
    step = r.plan.steps[0]
    p = override_path(ws.root, step)
    p.parent.mkdir(parents=True)
    p.write_text("import re\nreturn re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')\n")
    result = r.run().results[0]
    assert result.status == "done" and result.via == "your fix" and r.llm.calls == []


def test_a_wrong_fix_body_falls_back_to_the_normal_way(tmp_path):
    llm = ScriptedLLM(BODY_OK)
    r, ws = runner(tmp_path, llm, ENGLISH_RECIPE)
    p = override_path(ws.root, r.plan.steps[0])
    p.parent.mkdir(parents=True)
    p.write_text("return text\n")
    result = r.run().results[0]
    assert result.status == "done" and result.via == "" and llm.kinds == ["normal"]


def test_lessons_are_put_into_the_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEAGENT_LESSONS", str(tmp_path / "global.json"))
    (tmp_path / "global.json").write_text(json.dumps({"lessons": ["Never reassign a parameter."]}))
    llm = ScriptedLLM(BODY_OK)
    r, ws = runner(tmp_path, llm, ENGLISH_RECIPE)
    (ws.root / ".codeagent").mkdir(parents=True, exist_ok=True)
    (ws.root / ".codeagent" / "lessons.json").write_text(json.dumps({"lessons": ["Write every step as its own line."]}))
    r.lessons = load_lessons(ws.root)
    r.run()
    user = llm.calls[0][1]["content"]
    assert "Rules learned from earlier mistakes" in user and "- Never reassign a parameter." in user
    assert "- Write every step as its own line." in user


def test_lessons_are_capped_and_missing_files_are_fine(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEAGENT_LESSONS", str(tmp_path / "none.json"))
    assert load_lessons(tmp_path) == []
    (tmp_path / ".codeagent").mkdir()
    (tmp_path / ".codeagent" / "lessons.json").write_text(json.dumps({"lessons": [f"rule {i}" for i in range(20)]}))
    got = load_lessons(tmp_path, limit=3)
    assert got == ["rule 17", "rule 18", "rule 19"]


@pytest.mark.parametrize("recipe,expected", [
    ("1. if x < 0: raise ValueError\n2. return x", "if x < 0: raise ValueError\nreturn x"),
    ("1. result = []\n2. For each item in cart:\n   - if item == 1:\n     - result.append(item)\n   - else:\n     - result.append(0)\n3. return result",
     "result = []\nfor item in cart:\n    if item == 1:\n        result.append(item)\n    else:\n        result.append(0)\nreturn result"),
    ("1. import re, decimal\n2. return 1", "import re, decimal\nreturn 1"),
])
def test_compile_recipe_turns_code_lines_into_a_body(recipe, expected):
    assert compile_recipe("Create f(x).\n" + recipe + "\n") == expected


@pytest.mark.parametrize("recipe", [
    "1. Call check_args(price, quantity).\n2. return x",             # English
    "1. For each item in cart:\n2. return 1",                         # a header with nothing under it
    "Just words, no steps at all.",
    "",
])
def test_compile_recipe_gives_up_on_anything_that_is_not_all_code(recipe):
    assert compile_recipe(recipe) is None


def test_the_run_summary_says_which_steps_needed_help(tmp_path):
    r, _ = runner(tmp_path, ScriptedLLM(BODY_BAD), CODE_RECIPE)
    assert "rescued: recipe" in r.run().summary()
