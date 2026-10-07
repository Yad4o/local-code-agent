import os
import signal
import time

import pytest

from codeagent import __version__
from codeagent.cli import _Stopper, main
from codeagent.llm import LLMError
from codeagent.plan import parse_plan, template_text
from tests.test_runner import CORE_BAD, CORE_OK, INIT, TEST_CORE, FakeGuard, FakeLLM, isolated_git  # noqa: F401


def plan_toml(workspace=None, test_command="python -m pytest -q"):
    ws = f'workspace = "{workspace}"\n' if workspace else ""
    return f'''
title = "Slugify utility"
test_command = "{test_command}"
{ws}
[[step]]
title = "core function"
description = "Create slugify(text) in slugify/core.py. 'Hello, World!' -> 'hello-world'."
commit = "feat: add slugify core function"
tests = ["tests/test_core.py"]
files = ["slugify/__init__.py", "slugify/core.py"]
'''


@pytest.fixture
def plan(tmp_path):
    p = tmp_path / "plan.toml"
    p.write_text(plan_toml())
    return p


@pytest.fixture
def cfgfile(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("step_cooldown_s = 0\nmax_attempts = 2\ntest_timeout = 60\npoll_seconds = 0.1\n")
    return p


@pytest.fixture
def lines():
    return []


def run_cli(argv, lines, llm=None, **kw):
    kw.setdefault("guard_factory", lambda c, root: FakeGuard())
    kw.setdefault("llm_factory", lambda c: llm if llm is not None else FakeLLM())
    return main(argv, out=lines.append, **kw)


def text(lines):
    return "\n".join(str(l) for l in lines)


# --------------------------------------------------------------------- basics

def test_version(lines, capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"], out=lines.append)
    assert e.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_is_an_error(capsys):
    with pytest.raises(SystemExit) as e:
        main([], out=print)
    assert e.value.code == 2


# --------------------------------------------------------------------- template

def test_template_prints_a_valid_plan(lines):
    assert run_cli(["template"], lines) == 0
    parse_plan(text(lines))


def test_template_out_writes_file_and_refuses_to_overwrite(tmp_path, lines):
    out = tmp_path / "my.toml"
    assert run_cli(["template", "--out", str(out)], lines) == 0
    assert out.read_text() == template_text()
    out.write_text("precious")
    assert run_cli(["template", "--out", str(out)], lines) == 2
    assert out.read_text() == "precious"
    assert run_cli(["template", "--out", str(out), "--force"], lines) == 0
    assert out.read_text() == template_text()


# --------------------------------------------------------------------- check

def test_check_valid_plan(plan, lines, cfgfile):
    assert run_cli(["check", str(plan), "--config", str(cfgfile)], lines) == 0
    t = text(lines)
    assert "Slugify utility" in t and "core function" in t and "tests/test_core.py" in t


def test_check_reports_every_problem(tmp_path, lines):
    bad = tmp_path / "bad.toml"
    bad.write_text('title = ""\n')
    assert run_cli(["check", str(bad)], lines) == 2
    assert "title" in text(lines) and "step" in text(lines)


def test_check_missing_file(tmp_path, lines):
    assert run_cli(["check", str(tmp_path / "nope.toml")], lines) == 2
    assert "not found" in text(lines)


def test_check_rejects_a_test_command_that_is_not_allowed(tmp_path, lines):
    p = tmp_path / "p.toml"
    p.write_text(plan_toml(test_command="make test"))
    assert run_cli(["check", str(p)], lines) == 2
    assert "allowed" in text(lines)


def test_bad_config_file_is_a_clean_error(plan, tmp_path, lines):
    c = tmp_path / "bad.toml"
    c.write_text("nonsense_key = 1\n")
    assert run_cli(["check", str(plan), "--config", str(c)], lines) == 2
    assert "nonsense_key" in text(lines)


# --------------------------------------------------------------------- run

def test_run_success(plan, tmp_path, lines, cfgfile):
    ws = tmp_path / "proj"
    code = run_cli(["run", str(plan), "--workspace", str(ws), "--config", str(cfgfile)], lines,
                   llm=FakeLLM(TEST_CORE, INIT + CORE_OK))
    assert code == 0
    assert "DONE" in text(lines)
    assert (ws / "slugify" / "core.py").is_file()
    assert (ws / ".codeagent" / "agent.log").is_file()           # everything is logged to a file too
    assert "core function" in (ws / ".codeagent" / "agent.log").read_text()


def test_run_failure_exit_code(plan, tmp_path, lines, cfgfile):
    code = run_cli(["run", str(plan), "--workspace", str(tmp_path / "p"), "--config", str(cfgfile)], lines,
                   llm=FakeLLM(TEST_CORE, INIT + CORE_BAD, INIT + CORE_BAD))
    assert code == 1
    assert "FAILED" in text(lines)


def test_run_paused_exit_code(plan, tmp_path, lines, cfgfile):
    code = run_cli(["run", str(plan), "--workspace", str(tmp_path / "p"), "--config", str(cfgfile)], lines,
                   llm=FakeLLM(LLMError("cannot reach the model server")))
    assert code == 3
    assert "PAUSED" in text(lines)


def test_run_needs_a_workspace_from_somewhere(plan, lines, cfgfile):
    assert run_cli(["run", str(plan), "--config", str(cfgfile)], lines) == 2
    assert "workspace" in text(lines)


def test_run_uses_workspace_from_the_plan(tmp_path, lines, cfgfile):
    ws = tmp_path / "from_plan"
    p = tmp_path / "p.toml"
    p.write_text(plan_toml(workspace=ws))
    assert run_cli(["run", str(p), "--config", str(cfgfile)], lines,
                   llm=FakeLLM(TEST_CORE, INIT + CORE_OK)) == 0
    assert (ws / "slugify" / "core.py").is_file()


def test_cli_workspace_overrides_the_plan(tmp_path, lines, cfgfile):
    p = tmp_path / "p.toml"
    p.write_text(plan_toml(workspace=tmp_path / "from_plan"))
    cli_ws = tmp_path / "from_cli"
    run_cli(["run", str(p), "--workspace", str(cli_ws), "--config", str(cfgfile)], lines,
            llm=FakeLLM(TEST_CORE, INIT + CORE_OK))
    assert (cli_ws / "slugify" / "core.py").is_file()
    assert not (tmp_path / "from_plan").exists()


def test_dry_run_never_touches_the_model_or_disk(plan, tmp_path, lines, cfgfile):
    def boom(cfg):
        raise AssertionError("model must not be created in a dry run")

    ws = tmp_path / "proj"
    code = main(["run", str(plan), "--workspace", str(ws), "--config", str(cfgfile), "--dry-run"],
                out=lines.append, llm_factory=boom)
    assert code == 0
    assert not ws.exists()
    assert "core function" in text(lines)


# --------------------------------------------------------------------- doctor

def test_doctor_all_good(stub, lines, monkeypatch):
    monkeypatch.setenv("CODEAGENT_BASE_URL", stub.url)
    monkeypatch.setenv("CODEAGENT_MODEL", "qwen3.5:0.8b")
    stub.responses = [(200, {"models": [{"name": "qwen3.5:0.8b"}, {"name": "other:1b"}]})]
    assert main(["doctor"], out=lines.append) == 0
    t = text(lines)
    assert "[ok]" in t and "qwen3.5:0.8b" in t


def test_doctor_model_missing_lists_what_is_installed(stub, lines, monkeypatch):
    monkeypatch.setenv("CODEAGENT_BASE_URL", stub.url)
    monkeypatch.setenv("CODEAGENT_MODEL", "nonexistent:9b")
    stub.responses = [(200, {"models": [{"name": "qwen3.5:0.8b"}]})]
    assert main(["doctor"], out=lines.append) == 1
    t = text(lines)
    assert "[FAIL]" in t and "nonexistent:9b" in t and "qwen3.5:0.8b" in t


def test_doctor_matches_latest_tag(stub, lines, monkeypatch):
    monkeypatch.setenv("CODEAGENT_BASE_URL", stub.url)
    monkeypatch.setenv("CODEAGENT_MODEL", "mymodel")
    stub.responses = [(200, {"models": [{"name": "mymodel:latest"}]})]
    assert main(["doctor"], out=lines.append) == 0


def test_doctor_server_down(lines, monkeypatch):
    monkeypatch.setenv("CODEAGENT_BASE_URL", "http://127.0.0.1:9")      # nothing listens
    assert main(["doctor"], out=lines.append) == 1
    assert "[FAIL]" in text(lines) and "ollama serve" in text(lines)


# --------------------------------------------------------------------- smoke

GOOD_REPLY = "FILE: math_utils.py\n```python\ndef add(a, b):\n    return a + b\n```\n"


def test_smoke_passes_when_the_model_follows_the_format(stub, lines, monkeypatch):
    monkeypatch.setenv("CODEAGENT_BASE_URL", stub.url)
    stub.responses = [(200, {"message": {"content": GOOD_REPLY}})]
    assert main(["smoke", "--tries", "3"], out=lines.append) == 0
    assert "3/3" in text(lines)


def test_smoke_fails_when_the_model_ignores_the_format(stub, lines, monkeypatch):
    monkeypatch.setenv("CODEAGENT_BASE_URL", stub.url)
    stub.responses = [(200, {"message": {"content": "Sure! Here is some code: def add(a,b): return a+b"}})]
    assert main(["smoke", "--tries", "3"], out=lines.append) == 1
    assert "0/3" in text(lines)


# --------------------------------------------------------------------- daemon

def test_daemon_once_processes_the_inbox(tmp_path, lines, cfgfile):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    p = inbox / "slug.toml"
    p.write_text(plan_toml(workspace=tmp_path / "proj"))
    os.utime(p, (time.time() - 10,) * 2)
    code = run_cli(["daemon", "--inbox", str(inbox), "--once", "--config", str(cfgfile)], lines,
                   llm=FakeLLM(TEST_CORE, INIT + CORE_OK))
    assert code == 0
    assert (inbox / "done" / "slug.toml").is_file()
    assert (inbox / "agent.log").is_file()


# --------------------------------------------------------------------- signals

def test_first_signal_requests_a_graceful_stop_second_aborts():
    s = _Stopper()
    assert s() is False
    s._handle(signal.SIGTERM, None)
    assert s() is True
    with pytest.raises(KeyboardInterrupt):
        s._handle(signal.SIGTERM, None)


def test_stopper_restores_previous_handlers():
    before = signal.getsignal(signal.SIGTERM)
    s = _Stopper()
    s.install()
    assert signal.getsignal(signal.SIGTERM) == s._handle
    s.restore()
    assert signal.getsignal(signal.SIGTERM) == before


# --------------------------------------------------------------------- unix pipes

def test_closed_pipe_like_piping_into_head_exits_quietly():
    def closed_pipe(_line):
        raise BrokenPipeError

    assert main(["template"], out=closed_pipe) == 0


def test_console_entry_point_exits_with_the_command_status(monkeypatch, capsys):
    from codeagent.cli import entry

    monkeypatch.setattr("sys.argv", ["codeagent", "template"])
    with pytest.raises(SystemExit) as e:
        entry()
    assert e.value.code == 0
    assert "plan" in capsys.readouterr().out
