"""The whole stack, exactly as a user runs it: real `python -m agent` process ->
HTTP to a (fake) Ollama on loopback -> runner -> real pytest -> real git."""
import os
import signal
import subprocess
import sys
import time

import pytest

from tests.test_cli import plan_json
from tests.test_runner import CORE_OK, INIT, TEST_CORE


@pytest.fixture
def env(tmp_path, stub):
    gitcfg = tmp_path / "gitconfig"
    gitcfg.write_text("")
    return {**os.environ, "CODEAGENT_BASE_URL": stub.url, "CODEAGENT_MODEL": "fake:0.8b",
            "CODEAGENT_STEP_COOLDOWN_S": "0", "CODEAGENT_MAX_ATTEMPTS": "2",
            # machine-independent: never pause because of this host's RAM/battery/heat
            "CODEAGENT_MIN_FREE_MEM_MB": "0", "CODEAGENT_MIN_BATTERY_PCT": "0",
            "CODEAGENT_MAX_TEMP_C": "120", "CODEAGENT_MIN_FREE_DISK_MB": "0",
            "GIT_CONFIG_GLOBAL": str(gitcfg), "GIT_CONFIG_SYSTEM": os.devnull, "HOME": str(tmp_path)}


@pytest.fixture
def plan(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(plan_json())
    return p


def reply(text):
    return (200, {"message": {"role": "assistant", "content": text}, "done": True})


def agent(*args, env):
    return subprocess.run([sys.executable, "-m", "agent", *map(str, args)],
                          capture_output=True, text=True, env=env, timeout=120)


def test_run_builds_a_project_end_to_end(tmp_path, stub, env, plan):
    stub.responses = [reply(TEST_CORE), reply(INIT + CORE_OK)]
    ws = tmp_path / "proj"
    r = agent("run", plan, "--workspace", ws, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "DONE" in r.stdout and (ws / "slugify" / "core.py").is_file()
    log = subprocess.run(["git", "log", "--format=%s"], cwd=ws, capture_output=True, text=True).stdout
    assert log.strip() == "feat: add slugify core function"
    posts = [b for m, p, b in stub.requests if m == "POST"]
    assert len(posts) == 2
    assert all(p["model"] == "fake:0.8b" and p["think"] is True and p["stream"] is False for p in posts)


def test_thinking_can_be_switched_off_from_the_environment(tmp_path, stub, env, plan):
    stub.responses = [reply(TEST_CORE), reply(INIT + CORE_OK)]
    assert agent("run", plan, "--workspace", tmp_path / "p", env={**env, "CODEAGENT_THINK": "false"}).returncode == 0
    assert all(b["think"] is False for m, p, b in stub.requests if m == "POST")


def test_model_server_down_pauses_instead_of_failing(tmp_path, env, plan):
    r = agent("run", plan, "--workspace", tmp_path / "proj", env={**env, "CODEAGENT_BASE_URL": "http://127.0.0.1:9"})
    assert r.returncode == 3, r.stdout + r.stderr
    assert "PAUSED" in r.stdout and "cannot reach" in r.stdout
    assert not (tmp_path / "proj" / "slugify").exists()


def test_one_ctrl_c_stops_an_agent_that_is_waiting_for_resources(tmp_path, env, plan):
    """Regression: a single signal used to be ignored while waiting for RAM/heat to recover."""
    blocked = {**env, "CODEAGENT_MIN_FREE_MEM_MB": "1000000", "CODEAGENT_RESOURCE_WAIT_S": "600"}
    proc = subprocess.Popen([sys.executable, "-m", "agent", "run", str(plan), "--workspace", str(tmp_path / "proj")],
                            env=blocked, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(2.5)
    started = time.time()
    proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=15)
    assert time.time() - started < 8, "agent ignored the stop request while waiting"
    assert proc.returncode == 3, (proc.returncode, out, err)
    assert "PAUSED" in out and "stop requested" in out
