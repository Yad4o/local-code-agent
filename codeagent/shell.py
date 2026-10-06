"""Run commands the agent is allowed to run, without ever being able to hurt the laptop.

* no shell (so `;`, `&&`, `$()`, pipes are plain arguments)
* only allowlisted program names; `python`/`pytest` mean THIS interpreter
* wall-clock timeout that kills the whole process group (children too)
* memory (address space), CPU-time, and file-size limits (Linux)
* output is read through a bounded buffer: a runaway print loop can't eat RAM
* secrets (TOKEN/KEY/SECRET/PASSWORD env vars) are not passed through
"""
import os
import select
import shlex
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

try:
    import resource
except ImportError:  # non-POSIX
    resource = None

SECRET_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "APIKEY", "PRIVATE_KEY", "CREDENTIAL")
FLOOD_BYTES = 64 * 1024 * 1024        # kill a command that prints more than this
FILE_SIZE_LIMIT = 200 * 1024 * 1024    # no single file written by a command may exceed this


class CommandError(Exception):
    """The command was refused (not allowlisted / unparseable)."""


@dataclass
class CommandResult:
    command: str
    returncode: int
    output: str
    timed_out: bool = False
    flooded: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.flooded


def _parse(command: str, allowed: list[str]) -> list[str]:
    try:
        argv = shlex.split(command)
    except ValueError as e:
        raise CommandError(f"cannot parse command: {e}") from e
    if not argv:
        raise CommandError("empty command")
    prog = argv[0]
    if prog not in allowed:                      # exact name: no paths, no ./scripts
        raise CommandError(f"command not allowed: {prog!r} (allowed: {', '.join(allowed)})")
    if prog in ("python", "python3"):
        return [sys.executable, *argv[1:]]
    if prog == "pytest":
        return [sys.executable, "-m", "pytest", *argv[1:]]
    return argv


def _clean_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not any(m in k.upper() for m in SECRET_MARKERS)}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _limits(timeout: float, mem_mb: int):
    def apply():  # runs in the child just before exec
        if resource is None:
            return
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024,) * 2)
        cpu = int(timeout) + 5
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_SIZE_LIMIT,) * 2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return apply


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_command(command: str, cwd: str | Path, *, allowed: list[str], timeout: float,
                mem_mb: int, max_output_chars: int) -> CommandResult:
    argv = _parse(command, allowed)
    proc = subprocess.Popen(
        argv, cwd=str(cwd), env=_clean_env(),
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        start_new_session=True,                  # own process group so we can kill children
        preexec_fn=_limits(timeout, mem_mb) if resource else None,
    )

    keep = max_output_chars * 4                  # bytes of tail we hold in memory
    buf = bytearray()
    state = {"total": 0, "flooded": False}
    stop = threading.Event()
    fd = proc.stdout.fileno()

    def reader():
        # select + os.read (not a blocking buffered read) so that a child which
        # escaped the process group and keeps the pipe open can never hang us.
        while not stop.is_set():
            ready, _, _ = select.select([fd], [], [], 0.2)
            if not ready:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                return                            # EOF: every writer has exited
            state["total"] += len(chunk)
            buf.extend(chunk)
            if len(buf) > keep:
                del buf[:-keep]
            if state["total"] > FLOOD_BYTES:
                state["flooded"] = True
                _kill_group(proc)
                return

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    _kill_group(proc)                            # also reaps any leftover children
    proc.wait()
    t.join(timeout=2)                            # normally ends at EOF right away
    stop.set()                                   # otherwise tell it to give up
    t.join(timeout=2)
    proc.stdout.close()

    text = bytes(buf).decode("utf-8", errors="replace")
    if len(text) > max_output_chars:
        text = f"...[output truncated, showing last {max_output_chars} chars]\n" + text[-max_output_chars:]
    if timed_out:
        text += f"\n[timed out after {timeout:g}s and was killed]"
    if state["flooded"]:
        text += "\n[killed: command produced runaway output]"
    return CommandResult(command, proc.returncode, text, timed_out, state["flooded"])
