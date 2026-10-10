"""Tests for the hand-written glue. Put this file in <workspace>/tests/ and run: python -m pytest -q"""
from todo.cli import main
from todo.store import load_tasks


def run(tmp_path, *argv):
    lines = []
    code = main(["--file", str(tmp_path / "t.json"), *argv], out=lines.append)
    return code, lines


def test_add_then_list_shows_the_task_and_a_summary(tmp_path):
    assert run(tmp_path, "add", "milk") == (0, ["added [ ] 1 milk"])
    assert run(tmp_path, "list") == (0, ["[ ] 1 milk", "1 open, 0 done"])


def test_done_and_status_filter(tmp_path):
    run(tmp_path, "add", "milk")
    run(tmp_path, "add", "eggs")
    assert run(tmp_path, "done", "2") == (0, ["task 2 is done"])
    assert run(tmp_path, "list", "--status", "open") == (0, ["[ ] 1 milk", "1 open, 1 done"])
    assert run(tmp_path, "list", "--status", "done") == (0, ["[x] 2 eggs", "1 open, 1 done"])


def test_rm_removes_the_task_from_the_file(tmp_path):
    run(tmp_path, "add", "milk")
    assert run(tmp_path, "rm", "1") == (0, ["task 1 removed"])
    assert load_tasks(tmp_path / "t.json") == []


def test_unknown_id_and_empty_text_are_reported_not_raised(tmp_path):
    assert run(tmp_path, "done", "9") == (1, ["error: there is no task with id 9"])
    assert run(tmp_path, "add", "  ") == (1, ["error: the task text cannot be empty"])


def test_a_missing_file_is_an_empty_list(tmp_path):
    assert load_tasks(tmp_path / "nope.json") == []
