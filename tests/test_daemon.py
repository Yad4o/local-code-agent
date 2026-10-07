import os
import time

import pytest

from codeagent.config import Config
from codeagent.daemon import Daemon
from codeagent.llm import LLMError
from codeagent.state import WorkspaceLock
from tests.test_runner import CORE_BAD, CORE_OK, INIT, TEST_CORE, FakeGuard, FakeLLM, isolated_git  # noqa: F401


def plan_file(inbox, name, workspace, title="Slugify utility"):
    p = inbox / name
    p.write_text(f'''
title = "{title}"
test_command = "python -m pytest -q"
workspace = "{workspace}"

[[step]]
title = "core function"
description = "Create slugify(text) in slugify/core.py. 'Hello, World!' -> 'hello-world'."
commit = "feat: add slugify core function"
tests = ["tests/test_core.py"]
files = ["slugify/__init__.py", "slugify/core.py"]
''')
    old = time.time() - 10
    os.utime(p, (old, old))                        # "settled": not being written right now
    return p


@pytest.fixture
def cfg():
    return Config(step_cooldown_s=0, max_attempts=2, test_timeout=60, poll_seconds=0.1)


@pytest.fixture
def inbox(tmp_path):
    return tmp_path / "inbox"


def make_daemon(inbox, cfg, llm, **kw):
    kw.setdefault("settle_seconds", 0)
    return Daemon(cfg, inbox, llm_factory=lambda c: llm, guard_factory=lambda c, root: FakeGuard(),
                  log=lambda m: None, sleep=lambda s: None, **kw)


def names(d):
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


def test_prepare_creates_the_queue_folders(inbox, cfg):
    make_daemon(inbox, cfg, FakeLLM()).prepare()
    for sub in ("running", "done", "failed"):
        assert (inbox / sub).is_dir()


def test_successful_plan_is_built_and_filed_under_done(inbox, tmp_path, cfg):
    d = make_daemon(inbox, cfg, FakeLLM(TEST_CORE, INIT + CORE_OK))
    d.prepare()
    plan_file(inbox, "slug.toml", tmp_path / "proj")
    assert d.run_once() == 1
    assert (tmp_path / "proj" / "slugify" / "core.py").is_file()
    assert names(inbox / "done") == ["slug.report.txt", "slug.toml"]
    assert names(inbox) == ["done", "failed", "running"]               # inbox itself is empty
    assert "DONE" in (inbox / "done" / "slug.report.txt").read_text()


def test_plan_that_cannot_be_completed_goes_to_failed_with_a_report(inbox, tmp_path, cfg):
    d = make_daemon(inbox, cfg, FakeLLM(TEST_CORE, INIT + CORE_BAD, INIT + CORE_BAD))
    d.prepare()
    plan_file(inbox, "slug.toml", tmp_path / "proj")
    d.run_once()
    assert names(inbox / "failed") == ["slug.report.txt", "slug.toml"]
    assert "FAILED" in (inbox / "failed" / "slug.report.txt").read_text()


def test_invalid_plan_is_rejected_with_a_clear_error(inbox, cfg):
    d = make_daemon(inbox, cfg, FakeLLM())
    d.prepare()
    p = inbox / "bad.toml"
    p.write_text('title = ""\n')
    os.utime(p, (time.time() - 10,) * 2)
    d.run_once()
    assert names(inbox / "failed") == ["bad.error.txt", "bad.toml"]
    assert "title" in (inbox / "failed" / "bad.error.txt").read_text()


def test_plan_without_workspace_is_rejected(inbox, cfg):
    d = make_daemon(inbox, cfg, FakeLLM())
    d.prepare()
    p = plan_file(inbox, "nows.toml", "x")
    p.write_text(p.read_text().replace('workspace = "x"\n', ""))
    d.run_once()
    assert "workspace" in (inbox / "failed" / "nows.error.txt").read_text()


def test_paused_plan_returns_to_the_inbox_and_backs_off(inbox, tmp_path, cfg):
    d = make_daemon(inbox, cfg, FakeLLM(TEST_CORE, LLMError("cannot reach the model server")))
    d.prepare()
    plan_file(inbox, "slug.toml", tmp_path / "proj")
    d.run_once()
    assert "slug.toml" in names(inbox)                   # will be retried
    assert d.pause_delay > cfg.poll_seconds
    first = d.pause_delay
    d.llm_factory = lambda c: FakeLLM(LLMError("still down"))
    d.run_once()
    assert d.pause_delay > first                         # exponential backoff
    assert d.pause_delay <= 600


def test_backoff_resets_after_a_successful_plan(inbox, tmp_path, cfg):
    d = make_daemon(inbox, cfg, FakeLLM(LLMError("down")))
    d.prepare()
    plan_file(inbox, "slug.toml", tmp_path / "proj")
    d.run_once()
    assert d.pause_delay > cfg.poll_seconds
    d.llm_factory = lambda c: FakeLLM(TEST_CORE, INIT + CORE_OK)
    d.run_once()
    assert d.pause_delay == cfg.poll_seconds


def test_unexpected_crash_is_contained_and_next_plan_still_runs(inbox, tmp_path, cfg):
    calls = []

    def factory(c):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("kaboom")
        return FakeLLM(TEST_CORE, INIT + CORE_OK)

    d = make_daemon(inbox, cfg, None)
    d.llm_factory = factory
    d.prepare()
    a = plan_file(inbox, "a.toml", tmp_path / "proj_a")
    b = plan_file(inbox, "b.toml", tmp_path / "proj_b")
    os.utime(b, (time.time() - 5,) * 2)
    os.utime(a, (time.time() - 9,) * 2)
    d.run_once()
    assert "kaboom" in (inbox / "failed" / "a.error.txt").read_text()
    assert "Traceback" in (inbox / "failed" / "a.error.txt").read_text()
    assert (tmp_path / "proj_b" / "slugify" / "core.py").is_file()


def test_plans_left_in_running_after_a_crash_are_requeued_and_resumed(inbox, tmp_path, cfg):
    d = make_daemon(inbox, cfg, FakeLLM(TEST_CORE, INIT + CORE_OK))
    d.prepare()
    plan_file(inbox / "running", "slug.toml", tmp_path / "proj")      # simulates power loss
    d.prepare()
    assert "slug.toml" in names(inbox)
    d.run_once()
    assert "slug.toml" in names(inbox / "done")


def test_non_toml_files_and_folders_are_ignored(inbox, cfg):
    d = make_daemon(inbox, cfg, FakeLLM())
    d.prepare()
    (inbox / "notes.txt").write_text("hi")
    (inbox / "sub").mkdir()
    assert d.run_once() == 0
    assert (inbox / "notes.txt").exists()


def test_files_still_being_written_are_left_alone(inbox, tmp_path, cfg):
    d = make_daemon(inbox, cfg, FakeLLM(), settle_seconds=60)
    d.prepare()
    p = plan_file(inbox, "slug.toml", tmp_path / "proj")
    os.utime(p, None)                                        # modified just now
    assert d.run_once() == 0
    assert "slug.toml" in names(inbox)


def test_oldest_plan_goes_first(inbox, tmp_path, cfg):
    order = []

    def factory(c):
        order.append(1)
        return FakeLLM(TEST_CORE, INIT + CORE_OK)

    d = make_daemon(inbox, cfg, None)
    d.llm_factory = factory
    d.prepare()
    new = plan_file(inbox, "a_new.toml", tmp_path / "p_new", title="New")
    old = plan_file(inbox, "z_old.toml", tmp_path / "p_old", title="Old")
    os.utime(new, (time.time() - 2,) * 2)
    os.utime(old, (time.time() - 20,) * 2)
    d.run_once()
    assert (tmp_path / "p_old" / "slugify" / "core.py").stat().st_mtime <= \
           (tmp_path / "p_new" / "slugify" / "core.py").stat().st_mtime


def test_busy_workspace_is_not_failed_just_retried_later(inbox, tmp_path, cfg):
    d = make_daemon(inbox, cfg, FakeLLM())
    d.prepare()
    plan_file(inbox, "slug.toml", tmp_path / "proj")
    (tmp_path / "proj").mkdir()
    with WorkspaceLock(tmp_path / "proj"):
        d.run_once()
    assert "slug.toml" in names(inbox)
    assert names(inbox / "failed") == []


def test_stop_flag_halts_between_plans(inbox, tmp_path, cfg):
    stop = {"flag": False}
    d = make_daemon(inbox, cfg, FakeLLM(TEST_CORE, INIT + CORE_OK), should_stop=lambda: stop["flag"])
    d.prepare()
    plan_file(inbox, "a.toml", tmp_path / "pa")
    plan_file(inbox, "b.toml", tmp_path / "pb")
    stop["flag"] = True
    assert d.run_once() == 0


def test_name_collisions_in_done_do_not_overwrite(inbox, tmp_path, cfg):
    d = make_daemon(inbox, cfg, None)
    d.llm_factory = lambda c: FakeLLM(TEST_CORE, INIT + CORE_OK)
    d.prepare()
    plan_file(inbox, "slug.toml", tmp_path / "p1")
    d.run_once()
    plan_file(inbox, "slug.toml", tmp_path / "p2")
    d.run_once()
    assert len([n for n in names(inbox / "done") if n.endswith(".toml")]) == 2


def test_run_forever_loops_until_stopped(inbox, tmp_path, cfg):
    state = {"n": 0}

    def stop():
        state["n"] += 1
        return state["n"] > 6

    napped = []
    d = Daemon(cfg, inbox, llm_factory=lambda c: FakeLLM(), guard_factory=lambda c, r: FakeGuard(),
               log=lambda m: None, sleep=napped.append, settle_seconds=0, should_stop=stop)
    d.run_forever()
    assert napped                                         # it idled between polls
