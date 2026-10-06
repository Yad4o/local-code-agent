import os
import sys
import time

import pytest

from codeagent.shell import CommandError, run_command

ALLOWED = ["python", "python3", "pytest"]


def run(cmd, cwd, **kw):
    kw.setdefault("allowed", ALLOWED)
    kw.setdefault("timeout", 20)
    kw.setdefault("mem_mb", 1024)
    kw.setdefault("max_output_chars", 2000)
    return run_command(cmd, cwd, **kw)


def _is_dead(pid: int) -> bool:
    """Gone, or a zombie (killed but not yet reaped by its new parent)."""
    try:
        state = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return True
    return state in ("Z", "X")


def test_runs_allowed_command_and_captures_output(tmp_path):
    r = run('python -c "print(\'hello\')"', tmp_path)
    assert r.ok and r.returncode == 0
    assert r.output.strip() == "hello"
    assert not r.timed_out


def test_nonzero_exit_is_reported(tmp_path):
    r = run('python -c "import sys; print(\'bad\'); sys.exit(3)"', tmp_path)
    assert not r.ok and r.returncode == 3
    assert "bad" in r.output


def test_stderr_is_merged_into_output(tmp_path):
    r = run('python -c "import sys; sys.stderr.write(\'oops\')"', tmp_path)
    assert "oops" in r.output


@pytest.mark.parametrize(
    "cmd", ["rm -rf /", "curl http://example.com", "bash -c 'echo hi'", "/bin/ls", "./evil.sh",
            "git push", "pip install requests", "sudo python -c 1"],
)
def test_commands_outside_the_allowlist_never_run(tmp_path, cmd):
    with pytest.raises(CommandError, match="not allowed"):
        run(cmd, tmp_path)


@pytest.mark.parametrize("cmd", ["", "   ", "python 'unterminated"])
def test_empty_or_unparseable_command_rejected(tmp_path, cmd):
    with pytest.raises(CommandError):
        run(cmd, tmp_path)


def test_no_shell_so_metacharacters_are_just_arguments(tmp_path):
    marker = tmp_path / "pwned"
    script = tmp_path / "args.py"
    script.write_text("import sys; print(sys.argv[1:])\n")
    r = run(f"python args.py ; touch {marker} && echo $HOME `id`", tmp_path)
    assert not marker.exists()
    assert "';'" in r.output and "'&&'" in r.output


def test_python_means_the_agents_own_interpreter(tmp_path):
    r = run('python -c "import sys; print(sys.executable)"', tmp_path)
    assert r.output.strip() == sys.executable


def test_pytest_shortcut_runs_pytest_module(tmp_path):
    (tmp_path / "test_a.py").write_text("def test_a():\n    assert 1 + 1 == 2\n")
    r = run("pytest -q", tmp_path)
    assert r.ok and "1 passed" in r.output


def test_runs_in_the_given_cwd(tmp_path):
    r = run('python -c "import os; print(os.getcwd())"', tmp_path)
    assert os.path.realpath(r.output.strip()) == os.path.realpath(tmp_path)


def test_timeout_kills_the_process(tmp_path):
    start = time.time()
    r = run('python -c "import time; time.sleep(60)"', tmp_path, timeout=1)
    assert r.timed_out and not r.ok
    assert time.time() - start < 10


def test_timeout_also_kills_child_processes(tmp_path):
    pidfile = tmp_path / "child.pid"
    script = tmp_path / "parent.py"
    script.write_text(
        "import subprocess, sys, time\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(c.pid))\n"
        "time.sleep(60)\n"
    )
    r = run("python parent.py", tmp_path, timeout=2)
    assert r.timed_out
    pid = int(pidfile.read_text())
    time.sleep(0.5)
    assert _is_dead(pid), "child process survived the timeout"


def test_long_output_is_truncated_keeping_the_tail(tmp_path):
    r = run('python -c "print(\'A\'*5000); print(\'THE-END\')"', tmp_path, max_output_chars=300)
    assert len(r.output) < 600
    assert "truncated" in r.output
    assert "THE-END" in r.output


def test_runaway_output_is_stopped_and_memory_stays_bounded(tmp_path):
    script = tmp_path / "flood.py"
    script.write_text("import sys\nwhile True:\n    sys.stdout.write('x' * 100000)\n")
    start = time.time()
    r = run("python flood.py", tmp_path, timeout=30, max_output_chars=1000)
    assert not r.ok and r.flooded
    assert len(r.output) < 3000
    assert time.time() - start < 25


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="rlimit is Linux-specific")
def test_memory_limit_stops_a_memory_hog(tmp_path):
    r = run('python -c "b = bytearray(2 * 1024**3); print(len(b))"', tmp_path, mem_mb=512)
    assert not r.ok
    assert "MemoryError" in r.output


def test_secrets_are_not_passed_to_the_command(tmp_path, monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "ghp_secret")
    monkeypatch.setenv("MY_API_KEY", "k")
    monkeypatch.setenv("DB_PASSWORD", "p")
    monkeypatch.setenv("HARMLESS", "ok")
    r = run('python -c "import os; print(os.environ.get(\'GH_TOKEN\'), os.environ.get(\'MY_API_KEY\'), '
            'os.environ.get(\'DB_PASSWORD\'), os.environ.get(\'HARMLESS\'))"', tmp_path)
    assert r.output.strip() == "None None None ok"


def test_stdin_is_closed_so_input_does_not_hang(tmp_path):
    r = run('python -c "input()"', tmp_path, timeout=10)
    assert not r.ok and not r.timed_out
    assert "EOFError" in r.output
