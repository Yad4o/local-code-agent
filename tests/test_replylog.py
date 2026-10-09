
from agent import ReplyLog


def files(root):
    d = root / ".codeagent" / "replies"
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


def test_saves_raw_reply_with_a_readable_header(tmp_path):
    log = ReplyLog(tmp_path)
    p = log.save("core function", "tests", 2, "FILE: a.py\n```python\nx\n```", seconds=12.34,
                 truncated=True, temperature=0.45)
    text = p.read_text()
    assert "step: core function" in text
    assert "phase: tests" in text and "attempt: 2" in text
    assert "seconds: 12.3" in text
    assert "truncated: yes" in text
    assert "temperature: 0.45" in text
    assert text.endswith("FILE: a.py\n```python\nx\n```\n")
    assert p.parent == tmp_path / ".codeagent" / "replies"


def test_file_names_sort_in_chronological_order_and_say_what_they_are(tmp_path):
    log = ReplyLog(tmp_path)
    log.save("Core Function!", "tests", 1, "a", seconds=1, truncated=False, temperature=0.2)
    log.save("Core Function!", "impl", 3, "b", seconds=1, truncated=False, temperature=0.2)
    names = files(tmp_path)
    assert len(names) == 2
    assert "core-function" in names[0] and "tests" in names[0] and names[0].endswith("a1.txt")
    assert "impl" in names[1] and names[1].endswith("a3.txt")


def test_old_replies_are_pruned_so_the_disk_never_fills(tmp_path):
    log = ReplyLog(tmp_path, keep=5)
    for i in range(12):
        log.save("s", "tests", i + 1, f"reply {i}", seconds=0, truncated=False, temperature=0.2)
    names = files(tmp_path)
    assert len(names) == 5
    newest = (tmp_path / ".codeagent" / "replies" / names[-1]).read_text()
    assert "reply 11" in newest


def test_a_write_failure_never_crashes_the_agent(tmp_path):
    blocker = tmp_path / ".codeagent"
    blocker.write_text("i am a file, not a directory")
    assert ReplyLog(tmp_path).save("s", "tests", 1, "x", seconds=0, truncated=False, temperature=0.2) is None
