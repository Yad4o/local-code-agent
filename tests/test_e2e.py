"""The whole stack, exactly as a user runs it: real `codeagent` process ->
HTTP to a (fake) Ollama on loopback -> runner -> real pytest -> real git."""
import os
import subprocess
import sys
import time

import pytest

from tests.test_cli import plan_toml
from tests.test_runner import CORE_OK, INIT, TEST_CORE


@pytest.fixture
def env(tmp_path, stub):
    gitcfg = tmp_path / "gitconfig"
    gitcfg.write_text("")
    e = {**os.environ, "CODEAGENT_BASE_URL": stub.url, "CODEAGENT_MODEL": "fake:0.8b",
         "GIT_CONFIG_GLOBAL": str(gitcfg), "GIT_CONFIG_SYSTEM": os.devnull, "HOME": str(tmp_path)}
    return e


@pytest.fixture
def cfgfile(tmp_path):
    p = tmp_path / "c.toml"        # machine-independent: never pause because of this host's RAM/battery
    p.write_text("step_cooldown_s = 0\nmax_attempts = 2\nmin_free_mem_mb = 0\nmin_battery_pct = 0\n"
                 "max_temp_c = 120\nmin_free_disk_mb = 0\n")
    return p


def reply(text):
    return (200, {"message": {"role": "assistant", "content": text}, "done": True})


def codeagent(*args, env):
    return subprocess.run([sys.executable, "-m", "codeagent", *map(str, args)],
                          capture_output=True, text=True, env=env, timeout=120)


def test_run_builds_a_project_end_to_end(tmp_path, stub, env, cfgfile):
    stub.responses = [reply(TEST_CORE), reply(INIT + CORE_OK)]
    plan = tmp_path / "plan.toml"
    plan.write_text(plan_toml())
    ws = tmp_path / "proj"

    r = codeagent("run", plan, "--workspace", ws, "--config", cfgfile, env=env)

    assert r.returncode == 0, r.stdout + r.stderr
    assert "DONE" in r.stdout
    assert (ws / "slugify" / "core.py").is_file()
    log = subprocess.run(["git", "log", "--format=%s"], cwd=ws, capture_output=True, text=True).stdout
    assert log.strip() == "feat: add slugify core function"
    # the model was called exactly twice, over HTTP, with the right settings
    posts = [b for m, p, b in stub.requests if m == "POST"]
    assert len(posts) == 2
    assert all(p["model"] == "fake:0.8b" and p["think"] is False and p["stream"] is False for p in posts)
    assert posts[1]["messages"][1]["content"].count("assert slugify('Hello, World!')") == 1


def test_daemon_once_end_to_end(tmp_path, stub, env, cfgfile):
    stub.responses = [reply(TEST_CORE), reply(INIT + CORE_OK)]
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    p = inbox / "slug.toml"
    p.write_text(plan_toml(workspace=tmp_path / "proj"))
    os.utime(p, (time.time() - 10,) * 2)

    r = codeagent("daemon", "--inbox", inbox, "--once", "--config", cfgfile, env=env)

    assert r.returncode == 0, r.stdout + r.stderr
    assert (inbox / "done" / "slug.toml").is_file()
    assert "DONE" in (inbox / "done" / "slug.report.txt").read_text()
    assert (tmp_path / "proj" / "slugify" / "core.py").is_file()


def test_model_server_down_pauses_instead_of_failing(tmp_path, env, cfgfile):
    env = {**env, "CODEAGENT_BASE_URL": "http://127.0.0.1:9"}           # nothing listens
    plan = tmp_path / "plan.toml"
    plan.write_text(plan_toml())
    r = codeagent("run", plan, "--workspace", tmp_path / "proj", "--config", cfgfile, env=env)
    assert r.returncode == 3, r.stdout + r.stderr
    assert "PAUSED" in r.stdout and "cannot reach" in r.stdout
    assert not (tmp_path / "proj" / "slugify").exists()                  # nothing half-written


def test_ctrl_c_style_sigterm_stops_cleanly(tmp_path, stub, env, cfgfile):
    import signal

    stub.delay = 30                                  # the "model" is slow: we stop while waiting
    stub.responses = [reply(TEST_CORE)]
    plan = tmp_path / "plan.toml"
    plan.write_text(plan_toml())
    cfg = tmp_path / "c2.toml"
    cfg.write_text(cfgfile.read_text() + "request_timeout = 60\n")
    proc = subprocess.Popen([sys.executable, "-m", "codeagent", "run", str(plan), "--workspace",
                             str(tmp_path / "proj"), "--config", str(cfg)],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(2)
    proc.send_signal(signal.SIGTERM)                 # 1st signal: graceful
    time.sleep(0.5)
    proc.send_signal(signal.SIGTERM)                 # 2nd signal: abort now (model call is stuck)
    out, err = proc.communicate(timeout=20)
    assert proc.returncode in (130, 3), (proc.returncode, out, err)
    assert not (tmp_path / "proj" / "slugify").exists()
