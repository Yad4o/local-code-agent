from agent import Transcript
from tests.test_cli import lines, plan, run_cli  # noqa: F401
from tests.test_runner import CORE_OK, INIT, TEST_CORE, FakeLLM

MSGS = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "write add"}]


def test_transcript_keeps_prompt_and_reply_together_in_one_file(tmp_path):
    t = Transcript(tmp_path)
    t.prompt(MSGS, "core / tests / attempt 1")
    t.reply("FILE: a.py", seconds=1.5, truncated=False, temperature=0.2)
    body = t.path.read_text()
    assert t.path == tmp_path / ".codeagent" / "transcript.md"
    assert body.index("PROMPT: core / tests / attempt 1") < body.index("[system]") < body.index("be brief")
    assert body.index("write add") < body.index("MODEL ANSWER (raw)") < body.index("FILE: a.py")
    assert "seconds: 1.5" in body and "truncated: no" in body


def test_transcript_includes_thinking_when_it_was_streamed(tmp_path):
    t = Transcript(tmp_path)
    t.prompt(MSGS, "x")
    t.chunk("thinking", "let me ")
    t.chunk("thinking", "think")
    t.chunk("content", "ignored, the final reply is passed to reply()")
    t.reply("FILE: a.py", seconds=0, truncated=True, temperature=0.2)
    body = t.path.read_text()
    assert body.index("MODEL THINKING") < body.index("let me think") < body.index("MODEL ANSWER (raw)")
    assert "ignored" not in body and "truncated: yes" in body


def test_transcript_appends_every_call_and_never_crashes(tmp_path):
    t = Transcript(tmp_path)
    for n in (1, 2):
        t.prompt(MSGS, f"attempt {n}")
        t.reply(f"reply {n}", seconds=0, truncated=False, temperature=0.2)
    body = t.path.read_text()
    assert body.index("attempt 1") < body.index("reply 1") < body.index("attempt 2") < body.index("reply 2")
    (tmp_path / ".codeagent").chmod(0o500)
    try:
        t.prompt(MSGS, "unwritable")                      # must not raise
        t.reply("x", seconds=0, truncated=False, temperature=0.2)
    finally:
        (tmp_path / ".codeagent").chmod(0o700)


def test_run_saves_all_prompts_and_replies_without_live(plan, tmp_path, lines):  # noqa: F811
    ws = tmp_path / "proj"
    assert run_cli(["run", str(plan), "--workspace", str(ws)], lines, FakeLLM(TEST_CORE, INIT + CORE_OK)) == 0
    body = (ws / ".codeagent" / "transcript.md").read_text()
    assert body.count("==== PROMPT:") == 2 and body.count("==== MODEL ANSWER (raw) ====") == 2
    assert "Create slugify(text)" in body and "def test_basic" in body
    assert "==== PROMPT" not in (ws / ".codeagent" / "agent.log").read_text()
