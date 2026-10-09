import json
import signal

import pytest

from agent import TEMPLATE, LLMError, __version__, _Stopper, main
from tests.test_runner import (  # noqa: F401
    CORE_BAD,
    CORE_OK,
    INIT,
    TEST_CORE,
    FakeGuard,
    FakeLLM,
    isolated_git,
)


def plan_json(workspace=None, test_command="python -m pytest -q"):
    plan = {"title": "Slugify utility", "test_command": test_command, "steps": [{
        "title": "core function", "description": "Create slugify(text) in slugify/core.py.",
        "commit": "feat: add slugify core function", "tests": ["tests/test_core.py"],
        "files": ["slugify/__init__.py", "slugify/core.py"]}]}
    if workspace:
        plan["workspace"] = str(workspace)
    return json.dumps(plan)


@pytest.fixture
def plan(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(plan_json())
    return p


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setenv("CODEAGENT_STEP_COOLDOWN_S", "0.001")
    monkeypatch.setenv("CODEAGENT_MAX_ATTEMPTS", "2")


@pytest.fixture
def lines():
    return []


def run_cli(argv, lines, llm=None):
    return main(argv, out=lines.append, guard_factory=lambda c, root: FakeGuard(),
                llm_factory=lambda c: llm if llm is not None else FakeLLM())


def text(lines):
    return "\n".join(str(l) for l in lines)


def test_version_and_missing_command(lines, capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"], out=lines.append)
    assert e.value.code == 0 and __version__ in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        main([], out=print)
    assert e.value.code == 2


def test_template_prints_and_writes_without_overwriting(tmp_path, lines):
    assert run_cli(["template"], lines) == 0 and text(lines).strip() == TEMPLATE.strip()
    out = tmp_path / "my.json"
    assert run_cli(["template", "--out", str(out)], lines) == 0 and out.read_text() == TEMPLATE
    out.write_text("precious")
    assert run_cli(["template", "--out", str(out)], lines) == 2 and out.read_text() == "precious"
    assert run_cli(["template", "--out", str(out), "--force"], lines) == 0 and out.read_text() == TEMPLATE


def test_check(plan, tmp_path, lines):
    assert run_cli(["check", str(plan)], lines) == 0
    assert "Slugify utility" in text(lines) and "tests/test_core.py" in text(lines)
    bad = tmp_path / "bad.json"
    bad.write_text('{"title": ""}')
    lines.clear()
    assert run_cli(["check", str(bad)], lines) == 2 and "title" in text(lines)
    assert run_cli(["check", str(tmp_path / "nope.json")], lines) == 2
    bad.write_text(plan_json(test_command="make test"))
    lines.clear()
    assert run_cli(["check", str(bad)], lines) == 2 and "allowed" in text(lines)


def test_bad_environment_setting_is_a_clean_error(plan, lines, monkeypatch):
    monkeypatch.setenv("CODEAGENT_MAX_ATTEMPTS", "lots")
    assert run_cli(["check", str(plan)], lines) == 2 and "MAX_ATTEMPTS" in text(lines)


def test_run_success_writes_a_log_file(plan, tmp_path, lines):
    ws = tmp_path / "proj"
    assert run_cli(["run", str(plan), "--workspace", str(ws)], lines, FakeLLM(TEST_CORE, INIT + CORE_OK)) == 0
    assert "DONE" in text(lines) and (ws / "slugify" / "core.py").is_file()
    assert "core function" in (ws / ".codeagent" / "agent.log").read_text()


def test_run_exit_codes_for_failure_and_pause(plan, tmp_path, lines):
    assert run_cli(["run", str(plan), "--workspace", str(tmp_path / "a")], lines,
                   FakeLLM(TEST_CORE, INIT + CORE_BAD, INIT + CORE_BAD)) == 1
    assert "FAILED" in text(lines)
    assert run_cli(["run", str(plan), "--workspace", str(tmp_path / "b")], lines,
                   FakeLLM(LLMError("cannot reach the model server"))) == 3
    assert "PAUSED" in text(lines)


def test_workspace_comes_from_the_cli_or_the_plan(plan, tmp_path, lines):
    assert run_cli(["run", str(plan)], lines) == 2 and "workspace" in text(lines)
    p = tmp_path / "p.json"
    p.write_text(plan_json(workspace=tmp_path / "from_plan"))
    cli_ws = tmp_path / "from_cli"
    run_cli(["run", str(p), "--workspace", str(cli_ws)], lines, FakeLLM(TEST_CORE, INIT + CORE_OK))
    assert (cli_ws / "slugify" / "core.py").is_file() and not (tmp_path / "from_plan").exists()
    assert run_cli(["run", str(p)], lines, FakeLLM(TEST_CORE, INIT + CORE_OK)) == 0
    assert (tmp_path / "from_plan" / "slugify" / "core.py").is_file()


def test_dry_run_never_touches_the_model_or_disk(plan, tmp_path, lines):
    def boom(cfg):
        raise AssertionError("model must not be created in a dry run")

    ws = tmp_path / "proj"
    assert main(["run", str(plan), "--workspace", str(ws), "--dry-run"], out=lines.append, llm_factory=boom) == 0
    assert not ws.exists() and "core function" in text(lines)


def test_doctor(stub, lines, monkeypatch):
    monkeypatch.setenv("CODEAGENT_BASE_URL", stub.url)
    monkeypatch.setenv("CODEAGENT_MODEL", "mymodel")
    stub.responses = [(200, {"models": [{"name": "mymodel:latest"}, {"name": "other:1b"}]})]
    assert main(["doctor"], out=lines.append) == 0 and "[ok]" in text(lines)
    monkeypatch.setenv("CODEAGENT_MODEL", "nonexistent:9b")
    lines.clear()
    assert main(["doctor"], out=lines.append) == 1
    assert "[FAIL]" in text(lines) and "other:1b" in text(lines)


def test_doctor_server_down(lines, monkeypatch):
    monkeypatch.setenv("CODEAGENT_BASE_URL", "http://127.0.0.1:9")
    assert main(["doctor"], out=lines.append) == 1
    assert "ollama serve" in text(lines)


GOOD_REPLY = "FILE: math_utils.py\n```python\ndef add(a, b):\n    return a + b\n```\n"


@pytest.mark.parametrize("reply, code, score", [(GOOD_REPLY, 0, "3/3"), ("Sure! def add(a,b): return a+b", 1, "0/3")])
def test_smoke(stub, lines, monkeypatch, reply, code, score):
    monkeypatch.setenv("CODEAGENT_BASE_URL", stub.url)
    stub.responses = [(200, {"message": {"content": reply}})]
    assert main(["smoke", "--tries", "3"], out=lines.append) == code
    assert score in text(lines)


def test_first_signal_stops_gracefully_second_aborts_and_handlers_are_restored():
    s = _Stopper()
    before = signal.getsignal(signal.SIGTERM)
    s.install()
    assert signal.getsignal(signal.SIGTERM) == s._handle
    assert s() is False
    s._handle(signal.SIGTERM, None)
    assert s() is True
    with pytest.raises(KeyboardInterrupt):
        s._handle(signal.SIGTERM, None)
    s.restore()
    assert signal.getsignal(signal.SIGTERM) == before


def test_closed_pipe_exits_quietly_and_entry_point_returns_the_status(monkeypatch, capsys):
    def closed_pipe(_line):
        raise BrokenPipeError

    assert main(["template"], out=closed_pipe) == 0
    from agent import entry
    monkeypatch.setattr("sys.argv", ["codeagent", "template"])
    with pytest.raises(SystemExit) as e:
        entry()
    assert e.value.code == 0 and "steps" in capsys.readouterr().out
