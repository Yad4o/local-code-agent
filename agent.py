"""codeagent: an offline, resource-safe coding agent for small local models.

One module on purpose. Config, plan, prompts, log and the agent loop live here
until there is a real reason to split them. The harness decides what happens
next; the model only answers narrow questions.
"""
import argparse
import ast
import builtins
import fcntl
import hashlib
import ipaddress
import json
import logging
import logging.handlers
import math
import os
import re
import select
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path, PurePosixPath
from typing import Protocol
from urllib.parse import urlparse

__version__ = "0.1.0"


# ==========================================================================
# parsing
# Parse the model's reply. A tiny model follows a simple, rigid format far
# better than JSON/tool-calls:
# 
#     FILE: path/to/file.py
#     ```python
#     ...code...
#     ```
# ==========================================================================

HEADER = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?(?:file|path)\s*:\s*`?([^`*\n]+?)`?(?:\*\*)?\s*$", re.IGNORECASE
)
INLINE_PATH = re.compile(r"^\s*(?:#|//|--|/\*)\s*(?:file|path)\s*:\s*(\S+?)\s*(?:\*/)?\s*$", re.IGNORECASE)
FENCE_OPEN = re.compile(r"^\s*```[\w+.\-]*\s*$")
FENCE_CLOSE = re.compile(r"^\s*```\s*$")


@dataclass(frozen=True)
class FileBlock:
    path: str
    content: str


@dataclass
class ParseResult:
    blocks: list[FileBlock] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)      # things we quietly fixed (worth logging)
    anonymous: list[str] = field(default_factory=list)  # fenced blocks that came without any file name


def strip_thinking(text: str) -> str:
    """Remove Qwen-style <think>...</think> reasoning (closed, unclosed or half-open)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:                       # opening tag was cut off
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:                        # reasoning never finished
        text = text.split("<think>", 1)[0]
    return text.strip()


def _norm_path(p: str) -> str:
    p = p.strip()
    while p.startswith("./"):
        p = p[2:]
    return p


def _longest_valid_python_prefix(lines: list[str]) -> list[str] | None:
    """Drop trailing lines until what is left parses as Python (None if nothing does)."""
    for k in range(len(lines), 0, -1):
        chunk = lines[:k]
        if not any(l.strip() for l in chunk):
            return None
        try:
            ast.parse("\n".join(chunk))
        except SyntaxError:
            continue
        while chunk and not chunk[-1].strip():
            chunk = chunk[:-1]
        return chunk
    return None


def parse_file_blocks(text: str, *, truncated: bool = False) -> ParseResult:
    """`truncated=True` means the model hit its token limit, so an unclosed block was CUT OFF
    (repair it if it is Python). Otherwise an unclosed block just means it forgot the fence."""
    result = ParseResult()
    found: dict[str, str] = {}
    pending: str | None = None      # path waiting for its opening fence
    collecting = False              # inside a fenced block
    block_path: str | None = None   # path for the block being collected (None = anonymous)
    lines_acc: list[str] = []

    def finish(path: str | None, lines: list[str]) -> None:
        if path is None and lines:
            m = INLINE_PATH.match(lines[0])
            if m:
                path, lines = m.group(1), lines[1:]
        if path is None:
            if any(l.strip() for l in lines):
                result.anonymous.append("".join(l + "\n" for l in lines))
            return
        found[_norm_path(path)] = "".join(l + "\n" for l in lines)

    for line in strip_thinking(text).replace("\r\n", "\n").split("\n"):
        if collecting:
            if FENCE_CLOSE.match(line):
                finish(block_path, lines_acc)
                collecting, block_path, lines_acc = False, None, []
            else:
                lines_acc.append(line)
            continue
        if pending is not None:
            if not line.strip():
                continue
            if FENCE_OPEN.match(line):
                collecting, block_path, lines_acc, pending = True, pending, [], None
                continue
            pending = None                       # header without a fence: forget it
        m = HEADER.match(line)
        if m:
            pending = m.group(1)
        elif FENCE_OPEN.match(line):
            collecting, block_path, lines_acc = True, None, []

    if collecting and block_path is None and not truncated:
        finish(None, lines_acc)                  # an unclosed block with no name: still counts as anonymous
    if collecting and block_path is not None:
        path = _norm_path(block_path)
        if not truncated:
            finish(block_path, lines_acc)
            result.notes.append(f"{path}: the closing ``` was missing; accepted the code anyway")
        else:
            kept = _longest_valid_python_prefix(lines_acc) if path.endswith(".py") else None
            if kept:
                found[path] = "".join(l + "\n" for l in kept)
                result.notes.append(f"{path}: reply was cut off at the token limit; kept the first "
                                    f"{len(kept)} of {len(lines_acc)} lines that form valid Python")
            else:
                result.problems.append(
                    f"{path}: your reply was cut off because it was too long. Write a much SHORTER file: "
                    "no repeated code, only a few small functions, and close the code block with ```")

    result.blocks = [FileBlock(p, c) for p, c in found.items()]
    return result


# ==========================================================================
# plan
# Structured plan format (JSON). Strict on purpose: every plan looks the same,
# and mistakes are reported all at once, before the model runs anything.
# ==========================================================================

TOP_KEYS = {"title", "test_command", "workspace", "steps"}
STEP_KEYS = {"title", "description", "commit", "tests", "files", "context", "function", "examples", "signature"}
EXAMPLE_KEYS = {"args", "kwargs", "expect", "raises"}
FUNCTION_RE = re.compile(r"^([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*):([A-Za-z_]\w*)$")
FORBIDDEN_ROOTS = {".git", ".codeagent"}


class PlanError(Exception):
    def __init__(self, errors: list[str] | str):
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("\n".join(f"- {e}" for e in self.errors))


@dataclass(frozen=True, eq=False)
class Example:
    """One input -> output fact about the function, written by the human in the plan.
    The harness turns it into a real test; the model never gets to decide what is correct."""
    args: tuple = ()
    kwargs: dict | None = None
    expect: object = None
    raises: str | None = None           # name of a builtin exception; None means "returns `expect`"


@dataclass(frozen=True, eq=False)
class Step:
    title: str
    description: str
    commit: str
    tests: tuple[str, ...]
    files: tuple[str, ...]
    context: tuple[str, ...] = ()
    function: str | None = None         # "package.module:name": the function the examples call
    examples: tuple[Example, ...] = ()
    signature: str | None = None        # "name(a, b=1)": the harness writes the def line, the model only the body


@dataclass(frozen=True)
class Plan:
    title: str
    test_command: str
    workspace: str | None
    steps: tuple[Step, ...]


TEMPLATE = """\
{
  "title": "Slugify utility",
  "test_command": "python -m pytest -q",
  "steps": [
    {
      "title": "slugify lowercases and joins words with dashes",
      "description": "Create function slugify(text: str) -> str in slugify/core.py. Lowercase the text, replace each run of non-alphanumeric characters with a single '-', and strip leading/trailing '-'.",
      "commit": "feat: add slugify core function",
      "tests": ["tests/test_core.py"],
      "files": ["slugify/__init__.py", "slugify/core.py"],
      "function": "slugify.core:slugify",
      "signature": "slugify(text)",
      "examples": [
        {"args": ["Hello, World!"], "expect": "hello-world"},
        {"args": ["  A  B "], "expect": "a-b"},
        {"args": [123], "raises": "TypeError"}
      ]
    }
  ]
}
"""
# Every plan has this structure. Keep each step SMALL (one function) and list EVERY file it may touch.
# "function" + "examples" make the harness print the tests (the model never decides what is correct);
# "signature" makes the harness write the def line so the model only writes the body.
# Optional: "workspace" (top level), "context" (existing files to show the model, per step).


def plan_slug(plan: Plan) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", plan.title.lower()).strip("-")
    return slug or "plan"


def _nonempty_str(d: dict, key: str, where: str, errors: list[str]) -> str | None:
    v = d.get(key)
    if not isinstance(v, str) or not v.strip():
        errors.append(f"{where}: '{key}' is required and must be a non-empty string")
        return None
    return v.strip()


def _path_list(d: dict, key: str, where: str, errors: list[str], required: bool) -> tuple[str, ...]:
    if key not in d:
        if required:
            errors.append(f"{where}: '{key}' is required (a non-empty list of file paths)")
        return ()
    v = d[key]
    if not isinstance(v, list) or (required and not v) or not all(isinstance(p, str) for p in v):
        errors.append(f"{where}: '{key}' must be a {'non-empty ' if required else ''}list of strings")
        return ()
    out, seen = [], set()
    for raw in v:
        bad = _bad_path(raw)
        if bad:
            errors.append(f"{where}: bad path in '{key}' ({raw!r}): {bad}")
            continue
        if raw in seen:
            errors.append(f"{where}: duplicate path in '{key}': {raw}")
            continue
        seen.add(raw)
        out.append(raw)
    return tuple(out)


def _bad_path(p: str) -> str | None:
    if not p.strip():
        return "empty"
    if "\\" in p:
        return "use forward slashes"
    pp = PurePosixPath(p)
    if pp.is_absolute():
        return "must be relative to the workspace"
    if ".." in pp.parts:
        return "must not contain '..'"
    if pp.parts and pp.parts[0] in FORBIDDEN_ROOTS:
        return f"{pp.parts[0]}/ is reserved"
    return None


def _literal_problem(value, path: str = "value") -> str | None:
    """Only plain JSON data that prints as valid Python (no nan/inf)."""
    if isinstance(value, bool) or isinstance(value, (int, str)):
        return None
    if isinstance(value, float):
        return None if math.isfinite(value) else f"{path} is nan/inf, which cannot be written as a Python literal"
    if isinstance(value, list):
        return next((p for i, v in enumerate(value) if (p := _literal_problem(v, f"{path}[{i}]"))), None)
    if isinstance(value, dict):
        return next((p for k, v in value.items() if (p := _literal_problem(v, f"{path}[{k!r}]"))), None)
    return f"{path} has an unsupported type ({type(value).__name__}); use strings, numbers, booleans, lists, objects"


def _parse_examples(rs: dict, where: str, tests: tuple[str, ...], files: tuple[str, ...],
                    errors: list[str]) -> tuple[str | None, tuple[Example, ...]]:
    function, raw = rs.get("function"), rs.get("examples")
    if raw is None and function is None:
        return None, ()
    if raw is None or function is None:
        errors.append(f"{where}: 'function' and 'examples' go together (give both or neither)")
        return None, ()
    ok = True
    m = FUNCTION_RE.match(function) if isinstance(function, str) else None
    if not m:
        errors.append(f"{where}: 'function' must look like 'package.module:name', got {function!r}")
        ok = False
    elif m.group(1).replace(".", "/") + ".py" not in files:
        errors.append(f"{where}: 'function' module file {m.group(1).replace('.', '/')}.py must be listed in 'files'")
        ok = False
    if len(tests) != 1:
        errors.append(f"{where}: a step with examples needs exactly one path in 'tests' (the generated test file)")
        ok = False
    if not isinstance(raw, list) or not raw or not all(isinstance(e, dict) for e in raw):
        errors.append(f"{where}: 'examples' must be a non-empty list of objects")
        return None, ()
    out = []
    for j, e in enumerate(raw, 1):
        ew = f"{where}, example {j}"
        for k in sorted(set(e) - EXAMPLE_KEYS):
            errors.append(f"{ew}: unknown key '{k}'")
        args, kwargs = e.get("args", []), e.get("kwargs", {})
        if not isinstance(args, list):
            errors.append(f"{ew}: 'args' must be a list")
            ok = False
        if not isinstance(kwargs, dict) or not all(k.isidentifier() for k in kwargs):
            errors.append(f"{ew}: 'kwargs' must be an object whose keys are valid parameter names")
            ok = False
        if ("expect" in e) == ("raises" in e):
            errors.append(f"{ew}: give exactly one of 'expect' (the result) or 'raises' (an exception name)")
            ok = False
        raises = e.get("raises")
        if raises is not None:
            exc = getattr(builtins, raises, None) if isinstance(raises, str) else None
            if not (isinstance(exc, type) and issubclass(exc, BaseException)):
                errors.append(f"{ew}: 'raises' must be a builtin exception name like ValueError, got {raises!r}")
                ok = False
        for key in ("args", "kwargs", "expect"):
            if key in e and (problem := _literal_problem(e[key], key)):
                errors.append(f"{ew}: {problem}")
                ok = False
        if ok:
            out.append(Example(tuple(args), dict(kwargs) or None, e.get("expect"), raises))
    return (function, tuple(out)) if ok else (None, ())


def _parse_signature(rs: dict, where: str, function: str | None, files: tuple[str, ...],
                     errors: list[str]) -> str | None:
    sig = rs.get("signature")
    if sig is None:
        return None
    if not isinstance(sig, str) or not sig.strip():
        errors.append(f"{where}: 'signature' must be a non-empty string like \"add_item(cart, name, price=1)\"")
        return None
    if not function:
        errors.append(f"{where}: 'signature' needs 'function' and 'examples' (the tests it must satisfy)")
        return None
    module, name = function.split(":")
    problem = signature_problem(sig.strip(), name)
    if problem:
        errors.append(f"{where}: 'signature' {problem}")
        return None
    mod_file = module.replace(".", "/") + ".py"
    allowed = {mod_file}
    parts = module.split(".")[:-1]
    allowed |= {"/".join(parts[:i]) + "/__init__.py" for i in range(1, len(parts) + 1)}
    stray = [f for f in files if f not in allowed]
    if stray:
        errors.append(f"{where}: with 'signature', 'files' may only be the module file and its package "
                      f"__init__.py files; remove: {', '.join(stray)}")
        return None
    return sig.strip()


def parse_plan(text: str) -> Plan:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise PlanError(f"invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise PlanError("the plan must be a JSON object")

    errors: list[str] = []
    for k in sorted(set(data) - TOP_KEYS):
        errors.append(f"unknown top-level key '{k}'")
    title = _nonempty_str(data, "title", "plan", errors)
    test_command = _nonempty_str(data, "test_command", "plan", errors)
    if test_command:
        try:
            shlex.split(test_command)
        except ValueError as e:
            errors.append(f"plan: 'test_command' is not valid shell syntax: {e}")
    workspace = data.get("workspace")
    if workspace is not None and (not isinstance(workspace, str) or not workspace.strip()):
        errors.append("plan: 'workspace' must be a non-empty string when present")
        workspace = None

    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        errors.append("plan: 'steps' must be a non-empty list")
        raw_steps = []

    steps, titles = [], set()
    for i, rs in enumerate(raw_steps, 1):
        where = f"step {i}"
        if not isinstance(rs, dict):
            errors.append(f"{where}: must be an object")
            continue
        for k in sorted(set(rs) - STEP_KEYS):
            errors.append(f"{where}: unknown key '{k}'")
        st = _nonempty_str(rs, "title", where, errors)
        if st in titles:
            errors.append(f"{where}: duplicate step title '{st}'")
        titles.add(st)
        desc = _nonempty_str(rs, "description", where, errors)
        commit = _nonempty_str(rs, "commit", where, errors)
        if commit and ("\n" in commit or len(commit) > 72):
            errors.append(f"{where}: 'commit' must be a single line of at most 72 characters")
        tests = _path_list(rs, "tests", where, errors, required=True)
        files = _path_list(rs, "files", where, errors, required=True)
        context = _path_list(rs, "context", where, errors, required=False)
        both = set(tests) & set(files)
        if both:
            errors.append(f"{where}: path(s) listed in both 'tests' and 'files': {', '.join(sorted(both))}")
        function, examples = _parse_examples(rs, where, tests, files, errors)
        signature = _parse_signature(rs, where, function, files, errors)
        if st and desc and commit:
            steps.append(Step(st, desc, commit, tests, files, context, function, examples, signature))

    if errors:
        raise PlanError(errors)
    return Plan(title, test_command, workspace, tuple(steps))


def load_plan(path: str | Path) -> Plan:
    p = Path(path)
    if not p.is_file():
        raise PlanError(f"plan file not found: {p}")
    return parse_plan(p.read_text(encoding="utf-8"))


# ==========================================================================
# config
# Configuration: typed defaults + env overrides.
# 
# Defaults target a small laptop (2 GB GPU, ~12 GB RAM). Unknown keys and wrong
# types are errors so a typo in the config can never silently do nothing.
# ==========================================================================

class ConfigError(Exception):
    pass


@dataclass
class Config:
    # --- model backend ---
    backend: str = "ollama"                     # "ollama" or "openai" (llama.cpp server etc.)
    base_url: str = "http://localhost:11434"
    model: str = "qwen3.5-q4"                 # run `codeagent doctor` to list installed models
    think: bool = True                          # agentic coding needs the model to reason before it writes code
    think_level: str = "low"                   # low | medium | high; only models that support levels (gpt-oss) use it
    num_ctx: int = 16384                         # thinking tokens share the window with the prompt
    num_predict: int = 8192                     # max tokens per call, thinking included
    num_predict_tests: int = 8192               # smaller cap while writing tests
    num_predict_body: int = 8192                # cap when the model only writes a function body (skeleton mode)
    temperature: float = 0.2
    think_body: bool = False                    # False: a function body is written without thinking (a 0.8B model
                                                # that thinks about a 5-line body wanders for minutes and answers worse)
    temperature_no_think: float = 0.3           # temperature for calls that run without thinking
    repeat_penalty: float = 1.1                 # Ollama: discourages the "same sentence forever" loops
    top_p: float = 0.95                         # sampling cut-offs recommended for Qwen; 0 = leave to the server
    top_k: int = 20
    loop_guard: bool = True                     # stop a reply that is repeating itself (streamed calls only)
    max_think_chars: int = 8000                 # stop a call that thinks this long with no answer; 0 = unlimited
    retry_temperature_step: float = 0.05        # each retry runs this much hotter (small models get worse when hot)
    max_retry_temperature: float = 0.4          # retries never go hotter than this (or `temperature`, if higher)
    keep_alive: str = "30m"                     # keep model loaded between calls
    request_timeout: float = 900.0              # thinking makes replies slower
    allow_remote_llm: bool = False              # False = refuse any non-local base_url (offline guarantee)
    # --- agent loop ---
    max_attempts: int = 6                       # tries per phase (tests / implementation); calls without thinking are cheap
    max_test_rewrites: int = 2                  # times broken tests discovered while implementing are rewritten
    stop_on_failure: bool = True
    tidy_code: bool = True                      # strip docstrings/comments and re-print model code in one style
    use_ruff: bool = True                       # if `ruff` is installed: drop unused imports, catch undefined names
    step_cooldown_s: float = 2.0                # pause between model calls (laptop heat)
    # --- command / file limits ---
    allowed_commands: list[str] = field(default_factory=lambda: ["python", "python3", "pytest"])
    test_timeout: float = 120.0
    command_mem_mb: int = 2048                  # address-space cap for spawned commands
    max_output_chars: int = 3000                # tail of command output fed back to the model
    max_file_bytes: int = 100_000
    max_context_chars: int = 6000               # budget for code shown to the model
    # --- resource guard ---
    min_free_mem_mb: int = 1500
    min_battery_pct: int = 20                   # only enforced when running on battery
    max_temp_c: float = 88.0
    min_free_disk_mb: int = 1024
    resource_wait_s: float = 600.0              # how long to wait for resources before pausing


def _from_env(name: str, raw: str, default):
    try:
        if isinstance(default, bool):
            if raw.strip().lower() not in ("1", "true", "yes", "on", "0", "false", "no", "off"):
                raise ValueError
            return raw.strip().lower() in ("1", "true", "yes", "on")
        if isinstance(default, list):
            return [v.strip() for v in raw.split(",") if v.strip()]
        return type(default)(raw)
    except ValueError:
        raise ConfigError(f"CODEAGENT_{name.upper()}={raw!r} is not a valid {type(default).__name__}") from None


def load_config() -> Config:
    """Defaults, overridden by CODEAGENT_<SETTING> environment variables (e.g. CODEAGENT_MODEL,
    CODEAGENT_THINK=false, CODEAGENT_MAX_ATTEMPTS=6). No config file until one is needed."""
    cfg = Config()
    for f in fields(Config):
        raw = os.environ.get(f"CODEAGENT_{f.name.upper()}")
        if raw:
            setattr(cfg, f.name, _from_env(f.name, raw, getattr(cfg, f.name)))
    if cfg.backend not in ("ollama", "openai"):
        raise ConfigError(f"backend: must be 'ollama' or 'openai', got {cfg.backend!r}")
    if cfg.think_level not in ("low", "medium", "high"):
        raise ConfigError(f"think_level: must be low, medium or high, got {cfg.think_level!r}")
    return cfg


# ==========================================================================
# shell
# Run commands the agent is allowed to run, without ever being able to hurt the laptop.
# 
# * no shell (so `;`, `&&`, `$()`, pipes are plain arguments)
# * only allowlisted program names; `python`/`pytest` mean THIS interpreter
# * wall-clock timeout that kills the whole process group (children too)
# * memory (address space), CPU-time, and file-size limits (Linux)
# * output is read through a bounded buffer: a runaway print loop can't eat RAM
# * secrets (TOKEN/KEY/SECRET/PASSWORD env vars) are not passed through
# ==========================================================================

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


# ==========================================================================
# workspace
# Sandboxed file access. The agent can only touch files inside the workspace
# root, never .git/ or .codeagent/, never through symlinks that leave the root.
# ==========================================================================

RESERVED = {".git", ".codeagent"}


class WorkspaceError(Exception):
    pass


class Workspace:
    def __init__(self, root: str | Path, max_file_bytes: int):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_file_bytes = max_file_bytes

    def resolve(self, rel: str) -> Path:
        if not rel or not rel.strip():
            raise WorkspaceError("empty path")
        pp = PurePosixPath(rel)
        if pp.is_absolute():
            raise WorkspaceError(f"absolute path not allowed: {rel}")
        if ".." in pp.parts:
            raise WorkspaceError(f"'..' not allowed in path: {rel}")
        if pp.parts[0] in RESERVED:
            raise WorkspaceError(f"{pp.parts[0]}/ is reserved: {rel}")
        full = (self.root / pp).resolve()
        if not full.is_relative_to(self.root):
            raise WorkspaceError(f"path resolves outside the workspace: {rel}")
        return full

    def exists(self, rel: str) -> bool:
        return self.resolve(rel).is_file()

    def read(self, rel: str) -> str | None:
        p = self.resolve(rel)
        if not p.is_file():
            return None
        if p.stat().st_size > self.max_file_bytes:
            raise WorkspaceError(f"file too large to read: {rel}")
        try:
            return p.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise WorkspaceError(f"file is not valid UTF-8 text: {rel}") from e

    def write(self, rel: str, content: str) -> None:
        p = self.resolve(rel)
        if len(content.encode("utf-8")) > self.max_file_bytes:
            raise WorkspaceError(f"file too large to write: {rel}")
        p.parent.mkdir(parents=True, exist_ok=True)
        # atomic: a crash mid-write can never leave a half-written source file
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".tmp-", suffix=".part")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                f.write(content)
            os.replace(tmp, p)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def delete(self, rel: str) -> None:
        self.resolve(rel).unlink(missing_ok=True)

    def snapshot(self, rels: list[str] | tuple[str, ...]) -> dict[str, str | None]:
        return {r: self.read(r) for r in rels}

    def restore(self, snap: dict[str, str | None]) -> None:
        for rel, content in snap.items():
            if content is None:
                self.delete(rel)
            else:
                self.write(rel, content)


# ==========================================================================
# git
# Git helpers for unattended commits: never prompts, never hangs, never runs
# user hooks, and commits ONLY the files it is told to.
# ==========================================================================

GIT_TIMEOUT = 60
# kept out of `git status` via .git/info/exclude (never touches the user's .gitignore)
EXCLUDES = (".codeagent/", "__pycache__/", ".pytest_cache/")


class GitError(Exception):
    pass


def _run(root: str | Path, *args: str, check: bool = True) -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_PAGER": "cat"}
    try:
        p = subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True,
                           timeout=GIT_TIMEOUT, env=env, stdin=subprocess.DEVNULL)
    except FileNotFoundError as e:
        raise GitError("git is not installed or not on PATH") from e
    except (subprocess.TimeoutExpired, NotADirectoryError, OSError) as e:
        raise GitError(f"git {' '.join(args[:2])} failed: {e}") from e
    if check and p.returncode != 0:
        raise GitError(f"git {' '.join(args[:2])} failed: {(p.stderr or p.stdout).strip()}")
    return p.stdout.strip()


def ensure_repo(root: str | Path) -> bool:
    """Make `root` a git repo (branch main). True if it had to be created."""
    root = Path(root)
    created = not (root / ".git").exists()
    if created:
        _run(root, "init", "-q", "-b", "main")
    exclude = root / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    current = exclude.read_text() if exclude.exists() else ""
    have = set(current.splitlines())
    missing = [e for e in EXCLUDES if e not in have]
    if missing:
        sep = "" if current.endswith("\n") or not current else "\n"
        exclude.write_text(current + sep + "\n".join(missing) + "\n")
    return created


def current_branch(root: str | Path) -> str:
    return _run(root, "symbolic-ref", "--short", "HEAD")


def checkout_branch(root: str | Path, name: str) -> None:
    exists = subprocess.run(["git", "show-ref", "--verify", "--quiet", f"refs/heads/{name}"],
                            cwd=str(root)).returncode == 0
    _run(root, "checkout", "-q", name) if exists else _run(root, "checkout", "-q", "-b", name)


def _has_identity(root: str | Path) -> bool:
    return bool(_run(root, "config", "user.name", check=False)) and \
        bool(_run(root, "config", "user.email", check=False))


def commit_paths(root: str | Path, paths: list[str] | tuple[str, ...], message: str) -> str | None:
    """Commit exactly `paths`. Returns the short hash, or None if nothing changed."""
    _run(root, "add", "--", *paths)
    if not _run(root, "status", "--porcelain", "--", *paths):
        return None
    cfg = ["-c", "commit.gpgsign=false"]              # a signing prompt would hang forever
    if not _has_identity(root):
        cfg += ["-c", "user.name=codeagent", "-c", "user.email=codeagent@localhost"]
    _run(root, *cfg, "commit", "-q", "--no-verify", "-m", message, "--", *paths)
    return _run(root, "rev-parse", "--short", "HEAD")


# ==========================================================================
# state
# Crash-safe progress tracking + a lock so two agents never edit one workspace.
# ==========================================================================

DIR = ".codeagent"


class WorkspaceBusy(Exception):
    pass


def fingerprint(step: Step) -> str:
    """Changes whenever the step's definition changes, so edited steps are re-run."""
    return hashlib.sha256(json.dumps(asdict(step), sort_keys=True).encode()).hexdigest()[:16]


class RunState:
    def __init__(self, root: str | Path):
        self.dir = Path(root) / DIR
        self.path = self.dir / "state.json"
        self.data = self._load()

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("plans"), dict):
                return data
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            try:                                   # keep the broken file for inspection
                self.path.replace(self.path.with_name("state.json.corrupt"))
            except OSError:
                pass
        return {"version": 1, "plans": {}}

    def is_done(self, plan: Plan, step: Step) -> bool:
        rec = self.data["plans"].get(plan.title, {}).get(step.title)
        return bool(rec) and rec.get("fingerprint") == fingerprint(step)

    def mark_done(self, plan: Plan, step: Step, commit: str | None) -> None:
        self.data["plans"].setdefault(plan.title, {})[step.title] = {
            "fingerprint": fingerprint(step), "commit": commit,
        }
        self._save()

    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=".state-", suffix=".part")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())               # survive a power cut mid-run
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


class WorkspaceLock:
    """Exclusive, non-blocking lock. Released automatically if the process dies."""

    def __init__(self, root: str | Path):
        self.path = Path(root) / DIR / "lock"
        self.fd: int | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self.fd)
            self.fd = None
            raise WorkspaceBusy(f"another codeagent is already working in {self.path.parent.parent}")
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None


# ==========================================================================
# replylog
# Keep every raw model reply on disk (newest N), so you can read exactly what
# the model wrote when a step fails. Writing here must never crash the agent.
# ==========================================================================

class ReplyLog:
    def __init__(self, root: str | Path, keep: int = 60):
        self.dir = Path(root) / ".codeagent" / "replies"
        self.keep = keep
        self.counter = 0

    def save(self, step: str, phase: str, attempt: int, text: str, *, seconds: float,
             truncated: bool, temperature: float) -> Path | None:
        slug = re.sub(r"[^a-z0-9]+", "-", step.lower()).strip("-")[:40] or "step"
        self.counter += 1
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{self.counter:03d}-{slug}-{phase}-a{attempt}.txt"
        header = (f"# step: {step}\n# phase: {phase}\n# attempt: {attempt}\n"
                  f"# seconds: {seconds:.1f}\n# truncated: {'yes' if truncated else 'no'}\n"
                  f"# temperature: {temperature:g}\n---- raw reply ----\n")
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            path = self.dir / name
            path.write_text(header + text.rstrip("\n") + "\n", encoding="utf-8")
            for old in sorted(self.dir.glob("*.txt"))[:-self.keep]:
                old.unlink(missing_ok=True)
            return path
        except OSError:
            return None


# ==========================================================================
# resources
# Resource guard: pause the agent instead of crashing the laptop.
# 
# Checks free RAM, battery (only when discharging), CPU temperature and free
# disk. Every source is optional: if a sensor can't be read it simply does not
# block (a missing sensor must never stop a 24/7 agent forever).
# ==========================================================================

POLL_SECONDS = 5.0


class ResourceGuard:
    def __init__(self, cfg: Config, workdir: str | Path, *,
                 meminfo: str | Path = "/proc/meminfo",
                 power_dir: str | Path = "/sys/class/power_supply",
                 thermal_dir: str | Path = "/sys/class/thermal"):
        self.cfg = cfg
        self.workdir = Path(workdir)
        self.meminfo = Path(meminfo)
        self.power_dir = Path(power_dir)
        self.thermal_dir = Path(thermal_dir)

    # --- readings ------------------------------------------------------------
    def free_memory_mb(self) -> int | None:
        try:
            for line in self.meminfo.read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
        except (OSError, ValueError, IndexError):
            pass
        return None

    def battery(self) -> tuple[int, bool] | None:
        """(percent, discharging) of the first battery, or None if there is none."""
        try:
            for d in sorted(self.power_dir.glob("BAT*")):
                cap = int((d / "capacity").read_text().strip())
                status = (d / "status").read_text().strip().lower()
                return cap, status == "discharging"
        except (OSError, ValueError):
            pass
        return None

    def max_temp_c(self) -> float | None:
        temps = []
        try:
            for f in self.thermal_dir.glob("thermal_zone*/temp"):
                try:
                    c = int(f.read_text().strip()) / 1000
                except (OSError, ValueError):
                    continue
                if 1 <= c <= 150:                 # ignore garbage readings
                    temps.append(c)
        except OSError:
            pass
        return max(temps) if temps else None

    def free_disk_mb(self) -> int | None:
        try:
            return shutil.disk_usage(self.workdir).free // (1024 * 1024)
        except OSError:
            return None

    # --- decision --------------------------------------------------------------
    def problems(self) -> list[str]:
        out = []
        mem = self.free_memory_mb()
        if mem is not None and mem < self.cfg.min_free_mem_mb:
            out.append(f"low memory: {mem} MB free (need {self.cfg.min_free_mem_mb} MB)")
        bat = self.battery()
        if bat and bat[1] and bat[0] < self.cfg.min_battery_pct:
            out.append(f"low battery: {bat[0]}% and discharging (need {self.cfg.min_battery_pct}%)")
        temp = self.max_temp_c()
        if temp is not None and temp > self.cfg.max_temp_c:
            out.append(f"high temperature: {temp:.0f}°C (limit {self.cfg.max_temp_c:.0f}°C)")
        disk = self.free_disk_mb()
        if disk is not None and disk < self.cfg.min_free_disk_mb:
            out.append(f"low disk space: {disk} MB free (need {self.cfg.min_free_disk_mb} MB)")
        return out

    def wait_until_ok(self, *, sleep: Callable[[float], None] = time.sleep,
                      now: Callable[[], float] = time.monotonic,
                      on_wait: Callable[[list[str]], None] | None = None,
                      should_stop: Callable[[], bool] | None = None) -> bool:
        """Block until resources are fine. False if they stayed bad past resource_wait_s,
        or if a stop was requested (checked every second, so Ctrl+C is never ignored)."""
        deadline = now() + self.cfg.resource_wait_s
        while True:
            if should_stop and should_stop():
                return False
            reasons = self.problems()
            if not reasons:
                return True
            if on_wait:
                on_wait(reasons)
            if now() >= deadline:
                return False
            for _ in range(int(POLL_SECONDS)):          # sleep in 1s slices
                if should_stop and should_stop():
                    return False
                sleep(1.0)


# ==========================================================================
# tidy
# Deterministic clean-up tools. The model writes the logic; code does the tidying.
# 
# * `tidy_python`: removes docstrings and comments and re-prints the code in one standard style
#   (stdlib `ast`, no dependencies). A tiny model's chatter never reaches your files.
# * `ruff_fix` / `ruff_problems`: optional (used only when `ruff` is installed). Removes unused imports
#   and reports real mistakes (undefined names, redefinitions, syntax errors) with exact line numbers.
# ==========================================================================

RUFF_FIX_RULES = "F401"                          # unused imports
RUFF_CHECK_RULES = "F821,F811,E9"                # undefined name, redefinition, syntax/IO errors


def _drop_docstring(body: list[ast.stmt]) -> None:
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        del body[0]


def tidy_python(source: str) -> str:
    """Docstrings and comments out, canonical formatting in. Returns the source unchanged if it
    does not parse (the caller's own checks report that)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _drop_docstring(node.body)
            if not node.body and not isinstance(node, ast.Module):
                node.body = [ast.Pass()]
    out = ast.unparse(ast.fix_missing_locations(tree))
    return out + "\n" if out else ""


def ruff_available() -> bool:
    return shutil.which("ruff") is not None


def _ruff(args: list[str], root: Path, timeout: float) -> str | None:
    try:
        proc = subprocess.run(["ruff", *args], cwd=str(root), capture_output=True, text=True,
                              timeout=timeout, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout


def _code_paths(paths: list[str]) -> list[str]:
    return [p for p in paths if p.endswith(".py")]


def ruff_fix(root: Path, paths: list[str], timeout: float = 20.0) -> None:
    """Remove unused imports. `__init__.py` is skipped: its imports are re-exports on purpose."""
    todo = [p for p in _code_paths(paths) if not p.endswith("__init__.py")]
    if todo and ruff_available():
        _ruff(["check", "--fix", "--select", RUFF_FIX_RULES, "--no-cache", "--quiet", *todo], root, timeout)


def ruff_problems(root: Path, paths: list[str], timeout: float = 20.0) -> list[str]:
    """Real mistakes found without running anything, one short line each."""
    todo = _code_paths(paths)
    if not todo or not ruff_available():
        return []
    out = _ruff(["check", "--select", RUFF_CHECK_RULES, "--output-format", "concise",
                 "--no-cache", "--quiet", *todo], root, timeout)
    return [line.strip() for line in (out or "").splitlines() if line.strip() and ": " in line]


# ==========================================================================
# skeleton
# Skeleton mode: the plan gives the exact signature, the harness writes everything except the body.
# 
# The model replies with a few body lines instead of a whole file, so there is far less to get wrong
# (no repeated headers, no re-declared imports, no wrong function name) and far less to wait for.
# ==========================================================================

BODY_STOP = ["\nFILE:"]               # the model has nothing else to write after the body


def module_path(step: Step) -> str:
    return step.function.split(":")[0].replace(".", "/") + ".py"


def _function_name(step: Step) -> str:
    return step.function.split(":")[1]


def signature_problem(signature: str, name: str) -> str | None:
    try:
        tree = ast.parse(f"def {signature}:\n    pass\n")
    except SyntaxError as e:
        return f"is not a valid function signature ({e.msg})"
    fn = tree.body[0]
    if not isinstance(fn, ast.FunctionDef) or fn.name != name:
        return f"must define the function '{name}'"
    return None


def init_source(step: Step, init_path: str) -> str:
    """`pkg/__init__.py` re-exports the function so `from pkg import name` works."""
    pkg = init_path[: -len("/__init__.py")].replace("/", ".") if "/" in init_path else ""
    module = step.function.split(":")[0]
    rest = module[len(pkg) + 1:] if pkg else module
    return f"from .{rest} import {_function_name(step)}\n"


def step_context(step: Step) -> tuple[str, ...]:
    """The plan's context files, plus any existing-looking .py path the description names: a step that says
    "use check_args from cart/checks.py" needs that file shown and imported even if the plan forgot to list it."""
    own = {*step.files, *step.tests}
    out = list(step.context)
    for p in re.findall(r"[\w./-]+\.py\b", step.description):
        if p not in out and p not in own and not _bad_path(p):
            out.append(p)
    return tuple(out)


def missing_named_files(step: Step, existing: dict[str, str | None]) -> list[str]:
    """Files the description names ("use check_args from cart/checks.py") that are not in the workspace.
    The model cannot call what it cannot see, so six attempts at the step would only burn time."""
    own = {*step.files, *step.tests}
    return [p for p in step_context(step) if p not in own and not (existing.get(p) or "").strip()]


def _top_level_names(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]


def context_imports(step: Step, sources: dict[str, str | None]) -> list[str]:
    """`from pkg.mod import name` for every function the step's description names from its context files:
    the harness writes the imports, the model only writes the body (and so cannot forget them)."""
    own = module_path(step) if step.function else ""
    out = []
    for path in step_context(step):
        src = sources.get(path)
        if not src or not path.endswith(".py") or path.endswith("__init__.py") or path == own:
            continue
        module = path[:-3].replace("/", ".")
        for name in _top_level_names(src):
            if re.search(rf"\b{re.escape(name)}\b", step.description) and name != (_function_name(step) if step.function else ""):
                out.append(f"from {module} import {name}")
    return out


def _with_imports(source: str, extra: list[str]) -> str:
    have = {ln.strip() for ln in source.splitlines()}
    add = [ln for ln in extra if ln not in have]
    return "\n".join(add) + "\n" + source if add else source


def assemble_module(step: Step, body: str, imports: list[str] | None = None) -> tuple[str | None, str | None]:
    """(source, None) or (None, problem). Accepts a bare body; also a whole `def` if the model wrote one,
    but then the planned signature is put back: the model does not get to change it."""
    name = _function_name(step)
    extra = imports or []
    text = textwrap.dedent(body).strip("\n")
    if not text.strip():
        return None, "the function body is empty; write the lines that go inside the function"
    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = None
    if tree:
        node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
        if node is not None:
            planned = ast.parse(f"def {step.signature}:\n    pass\n").body[0] if step.signature else None
            if isinstance(planned, ast.FunctionDef):
                node.args, node.returns, node.decorator_list = planned.args, planned.returns, []
                return _with_imports(ast.unparse(ast.fix_missing_locations(tree)) + "\n", extra), None
            return _with_imports(text + "\n", extra), None
        if any(isinstance(n, ast.AsyncFunctionDef) and n.name == name for n in tree.body):
            return _with_imports(text + "\n", extra), None
    lines = text.splitlines()
    i = 0
    while i < len(lines) and re.match(r"(import|from)\s+\S", lines[i]):
        i += 1                                           # leading imports belong at the top of the module
    imps, rest = lines[:i], lines[i:]
    imps = [*[e for e in extra if e not in imps], *imps]
    src = "\n".join([*imps, "", f"def {step.signature}:", *[("    " + l) if l.strip() else "" for l in rest], ""])
    try:
        ast.parse(src)
    except SyntaxError as e:
        return None, f"the body is not valid Python inside `def {step.signature}:` ({e.msg}, line {e.lineno} of the module)"
    return src, None


def assemble_files(step: Step, got: dict[str, str], existing_inits: dict[str, str | None] | None = None,
                   context_sources: dict[str, str | None] | None = None) -> tuple[dict[str, str], list[str]]:
    """Turn the model's body into the complete set of files the step owns. A package `__init__.py` that
    earlier steps already filled keeps its lines; this step's re-export is added to it."""
    path = module_path(step)
    source, problem = assemble_module(step, got.get(path, ""), context_imports(step, context_sources or {}))
    if problem:
        return {}, [problem]
    files = {path: source}
    for p in step.files:
        if p.endswith("__init__.py") and p != path:
            line = init_source(step, p)
            old = (existing_inits or {}).get(p) or ""
            files[p] = old if line.strip() in old.splitlines() else old + ("" if not old or old.endswith("\n") else "\n") + line
    return files, []


# ==========================================================================
# spec_tests
# Turn the human's examples into a test file, with no model involved.
# 
# The tiny model is an unreliable typist: when it also writes the tests it invents
# contradictory expectations, calls the function with the wrong arguments and repeats
# test names. So the plan carries the facts and this module prints them as pytest code.
# ==========================================================================

def _call(name: str, ex: Example) -> str:
    parts = [repr(a) for a in ex.args] + [f"{k}={v!r}" for k, v in (ex.kwargs or {}).items()]
    return f"{name}({', '.join(parts)})"


def render_tests(step: Step) -> str:
    module, name = step.function.split(":")
    lines = []
    if any(e.raises for e in step.examples):
        lines += ["import pytest"]
    lines += [f"from {module} import {name}", "", ""]
    for i, ex in enumerate(step.examples, 1):
        if ex.raises:
            lines += [f"def test_example_{i}_raises_{ex.raises}():",
                      f"    with pytest.raises({ex.raises}):",
                      f"        {_call(name, ex)}"]
        else:
            call = _call(name, ex)
            lines += [f"def test_example_{i}():",
                      f"    result = {call}",
                      f"    expected = {ex.expect!r}",
                      f"    assert result == expected, {call!r} + ' returned ' + repr(result) + ', but it must return ' + repr(expected)"]
        lines += ["", ""]
    source = "\n".join(lines).rstrip("\n") + "\n"
    ast.parse(source)                                   # a bug here must be loud, never silent
    return source


def defined_names(source: str) -> set[str]:
    """Names defined at the top level of a module: def, class, assignment, import."""
    names: set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return names
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(t.id for t in targets if isinstance(t, ast.Name))
    return names


def returns_a_value(source: str, name: str) -> bool | None:
    """Does function `name` have a path that returns (or yields) something? None = cannot tell."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    fn = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name), None)
    if fn is None:
        return None
    stack = list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue                                    # a nested function's return is not ours
        if isinstance(node, ast.Return) and node.value is not None \
                and not (isinstance(node.value, ast.Constant) and node.value.value is None):
            return True
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            return True
        stack.extend(ast.iter_child_nodes(node))
    return False


# ==========================================================================
# testcheck
# Catch broken test files before they poison a whole step.
# 
# A test that can never run (a parameter nothing provides, a syntax error, no tests
# at all) fails every implementation the same way, so retrying the implementation
# is pointless. These checks let the runner send the problem back to the test phase.
# ==========================================================================

BUILTIN_FIXTURES = frozenset({
    "tmp_path", "tmp_path_factory", "tmpdir", "tmpdir_factory", "monkeypatch", "capsys",
    "capsysbinary", "capfd", "capfdbinary", "caplog", "request", "recwarn", "pytestconfig",
    "cache", "record_property", "record_testsuite_property", "doctest_namespace",
})


def _is_decorator(node: ast.expr, name: str) -> bool:
    target = node.func if isinstance(node, ast.Call) else node
    return (isinstance(target, ast.Attribute) and target.attr == name) or \
           (isinstance(target, ast.Name) and target.id == name)


def _parametrized_names(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for dec in func.decorator_list:
        if not (isinstance(dec, ast.Call) and _is_decorator(dec, "parametrize") and dec.args):
            continue
        first = dec.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            names.update(n.strip() for n in first.value.split(","))
        elif isinstance(first, (ast.List, ast.Tuple)):
            names.update(e.value for e in first.elts
                         if isinstance(e, ast.Constant) and isinstance(e.value, str))
    return names


def _required_params(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Parameters pytest must supply: the ones without a default value."""
    a = func.args
    positional = [*a.posonlyargs, *a.args]
    required = [p.arg for p in positional[:len(positional) - len(a.defaults)]]
    required += [p.arg for p, d in zip(a.kwonlyargs, a.kw_defaults) if d is None]
    return [p for p in required if p not in ("self", "cls")]


def check_test_source(source: str, known_fixtures: frozenset[str] | set[str] = frozenset()) -> list[str]:
    """Problems that make a test file unrunnable. Syntax errors are reported elsewhere."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    defined = {f.name for f in funcs if any(_is_decorator(d, "fixture") for d in f.decorator_list)}
    available = BUILTIN_FIXTURES | defined | set(known_fixtures)
    tests = [f for f in funcs if f.name.startswith("test") and f.name not in defined]
    if not tests:
        return ["there is no test function; write functions named test_something()"]
    problems = []
    for scope in [tree, *[n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]]:
        seen: set[str] = set()
        for f in scope.body:
            if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name.startswith("test"):
                if f.name in seen:
                    problems.append(f"{f.name}() is defined twice; the second silently replaces the first. "
                                    f"Give every test its own name.")
                seen.add(f.name)
    for f in tests:
        missing = [p for p in _required_params(f) if p not in available | _parametrized_names(f)]
        if missing:
            problems.append(
                f"{f.name}() has parameter(s) {', '.join(missing)} that nothing provides, so pytest "
                f"errors before running it. Every test function must take NO parameters: "
                f"write the input value inside the test.")
    return problems


def diagnose_broken_tests(output: str) -> str | None:
    """Spot pytest output that is the tests' fault, not the implementation's."""
    m = re.search(r"fixture '([^']+)' not found", output)
    if m:
        return (f"pytest could not find fixture '{m.group(1)}'. Every test function must take NO "
                f"parameters: write the input value inside the test.")
    if re.search(r"^E\s+(SyntaxError|IndentationError|TabError)\b", output, re.MULTILINE):
        return "the test file has a syntax error; write plain, valid Python."
    if re.search(r"\bno tests ran\b", output):
        return "pytest found no tests; name each test function starting with test_."
    return None


# ==========================================================================
# llm
# Talk to the local model server (Ollama, or any OpenAI-compatible server such
# as llama.cpp). Standard library only. Refuses non-local URLs unless explicitly
# allowed, so code can never leave the machine by accident.
# ==========================================================================

class LLMError(Exception):
    pass


class NoAnswer(LLMError):
    """The model finished or was stopped without writing a usable answer (ran out of tokens, or looped)."""


class LoopGuard:
    """Watches a streamed reply and says when it is going round in circles, so the call can be cut short
    instead of burning all num_predict tokens. Catches small loops (a phrase again and again), big loops
    (a whole paragraph again), the same line coming back, and endless thinking with no answer."""

    WINDOW = 8000

    def __init__(self, max_think_chars: int = 0):
        self.max_think_chars = max_think_chars
        self.text = {"thinking": "", "content": ""}
        self.thinking_total = 0
        self.answering = False
        self._since = 0

    def feed(self, kind: str, piece: str) -> str | None:
        if kind not in self.text:
            return None
        self.text[kind] = (self.text[kind] + piece)[-self.WINDOW:]
        if kind == "content":
            self.answering = self.answering or bool(piece.strip())
        else:
            self.thinking_total += len(piece)
            if self.max_think_chars and not self.answering and self.thinking_total > self.max_think_chars:
                return f"thought for more than {self.max_think_chars} characters without answering"
        self._since += len(piece)
        if self._since < 60:
            return None
        self._since = 0
        if self._repeating_tail(self.text[kind], 4 if kind == "content" else 3):
            return "the model started repeating itself"
        if self._repeated_lines(self.text[kind], 12 if kind == "content" else 8):
            return "the model keeps writing the same line again and again (repeating itself)"
        return None

    @staticmethod
    def _repeating_tail(text: str, min_reps: int) -> bool:
        for p in range(8, len(text) // min_reps + 1):
            reps = max(min_reps, -(-100 // p))
            if len(text) < reps * p:
                continue
            unit = text[-p:]
            if len(set(unit.strip())) > 3 and text[-p * reps:] == unit * reps:
                return True
        return False

    @staticmethod
    def _repeated_lines(text: str, limit: int) -> bool:
        counts: dict[str, int] = {}
        for line in text.splitlines()[-80:]:
            line = line.strip()
            if len(line) >= 12 and len(set(line)) > 3:
                counts[line] = counts.get(line, 0) + 1
                if counts[line] >= limit:
                    return True
        return False


class Reply(str):
    """The model's text. `.truncated` is True if it hit the token limit (it was cut off).
    `.no_answer` is a message for the model when there was no usable answer at all."""

    truncated: bool = False
    no_answer: str = ""


def _no_answer_feedback(e: Exception) -> str:
    if "repeat" in str(e):
        return ("your last attempt got stuck repeating itself and was stopped. "
                "Do not repeat yourself: decide quickly, then write the answer.")
    return ("your last attempt ran out of time while thinking and wrote no answer. "
            "Think only a few short lines, then write the answer straight away.")


def _reply(text: str, truncated: bool) -> Reply:
    r = Reply(text)
    r.truncated = truncated
    return r


class LLMClient(Protocol):
    def chat(self, messages: list[dict], *, temperature: float | None = None,
             num_predict: int | None = None, stop: list[str] | None = None,
             on_token: Callable[[str, str], None] | None = None, think: bool | None = None) -> str: ...


def _check_url(cfg: Config) -> str:
    url = cfg.base_url.strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise LLMError(f"invalid URL {cfg.base_url!r} in base_url: expected http://host:port")
    if not cfg.allow_remote_llm:
        host = parsed.hostname
        try:
            is_local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            is_local = host.lower() == "localhost"
        if not is_local:
            raise LLMError(
                f"base_url {cfg.base_url!r} is not a local address. codeagent is offline-only by "
                "default; set allow_remote_llm = true in the config if you really want this."
            )
    return url


class _BaseClient:
    def __init__(self, cfg: Config, *, sleep: Callable[[float], None] = time.sleep, retries: int = 2):
        self.cfg = cfg
        self.base = _check_url(cfg)
        self.sleep = sleep
        self.retries = retries

    def _request(self, path: str, payload: dict | None,
                 on_chunk: Callable[[str, str], None] | None = None, guard: LoopGuard | None = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            self.base + path, data=data, method="POST" if payload is not None else "GET",
            headers={"Content-Type": "application/json"},
        )
        last: LLMError | None = None
        for attempt in range(self.retries + 1):
            if attempt:
                self.sleep(2.0 * attempt)                 # backoff: 2s, 4s, ...
            try:
                with urllib.request.urlopen(req, timeout=self.cfg.request_timeout) as resp:
                    if on_chunk:
                        return self._read_stream(resp, on_chunk, guard)
                    raw = resp.read()
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:300]
                if e.code == 404 and "model" in body.lower():
                    raise LLMError(
                        f"model {self.cfg.model!r} not found on the server. Run "
                        "`codeagent doctor` to list installed models, or `ollama pull <name>`."
                    ) from e
                last = LLMError(f"model server returned HTTP {e.code}: {body}")
                if e.code >= 500:
                    continue                              # server busy/loading: retry
                raise last from e
            except TimeoutError as e:
                raise LLMError(f"request timed out after {self.cfg.request_timeout:g}s") from e
            except urllib.error.URLError as e:
                if isinstance(e.reason, (socket.timeout, TimeoutError)):
                    raise LLMError(f"request timed out after {self.cfg.request_timeout:g}s") from e
                last = LLMError(
                    f"cannot reach the model server at {self.base} ({e.reason}). "
                    "Is it running? For Ollama: `ollama serve`."
                )
                continue
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                raise LLMError(f"model server sent invalid JSON: {raw[:200]!r}") from e
        raise last  # type: ignore[misc]

    @staticmethod
    def _read_stream(resp, on_chunk: Callable[[str, str], None], guard: LoopGuard | None = None) -> dict:
        """Read Ollama's line-by-line stream, show each piece as it arrives, return the usual one-piece reply.
        If the guard says the model is looping, stop reading: closing the connection stops the generation."""
        content, thinking, last, aborted = [], [], {}, None
        for line in resp:
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as e:
                raise LLMError(f"model server sent invalid JSON: {line[:200]!r}") from e
            if chunk.get("error"):
                raise LLMError(f"model server error: {chunk['error']}")
            msg = chunk.get("message") or {}
            for kind, store in (("thinking", thinking), ("content", content)):
                if msg.get(kind):
                    store.append(msg[kind])
                    on_chunk(kind, msg[kind])
                    aborted = guard.feed(kind, msg[kind]) if guard else None
                    if aborted:
                        break
            last = chunk
            if aborted:
                break
        return {"message": {"role": "assistant", "content": "".join(content), "thinking": "".join(thinking)},
                "done_reason": "length" if aborted else last.get("done_reason"), "aborted": aborted,
                "eval_count": last.get("eval_count"), "done": last.get("done")}

    @staticmethod
    def _finish(text, truncated: bool = False) -> Reply:
        if not isinstance(text, str):
            raise LLMError("unexpected response shape from the model server")
        text = strip_thinking(text)
        if not text.strip():
            raise LLMError("model returned an empty reply")
        return _reply(text, truncated)


def think_value(cfg: Config):
    """Ollama's `think`: on/off, or low/medium/high for models that take a level (gpt-oss)."""
    if not cfg.think:
        return False
    return cfg.think_level if cfg.model.lower().startswith("gpt-oss") else True


def _cut_at_stop(text: str, stop: list[str] | None) -> str:
    for marker in stop or []:
        text = text.split(marker, 1)[0]
    return text


class OllamaClient(_BaseClient):
    def chat(self, messages: list[dict], *, temperature: float | None = None,
             num_predict: int | None = None, stop: list[str] | None = None,
             on_token: Callable[[str, str], None] | None = None, think: bool | None = None) -> Reply:
        c = self.cfg
        think = False if think is False else think_value(c)       # think=False: this one call answers without thinking
        options = {"num_ctx": c.num_ctx,
                   "num_predict": num_predict if num_predict is not None else c.num_predict,
                   "temperature": temperature if temperature is not None else c.temperature,
                   "repeat_penalty": c.repeat_penalty}
        if c.top_p:
            options["top_p"] = c.top_p
        if c.top_k:
            options["top_k"] = c.top_k
        guard = LoopGuard(c.max_think_chars) if (c.loop_guard and on_token) else None
        if stop and not think:
            options["stop"] = stop      # while thinking, the model's reasoning can contain the marker and end the reply early
        data = self._request("/api/chat", {
            "model": c.model, "messages": messages, "stream": on_token is not None,
            "think": think,
            "keep_alive": c.keep_alive,
            "options": options,
        }, on_chunk=on_token, guard=guard)
        try:
            msg = data["message"]
            content = _cut_at_stop(str(msg.get("content") or ""), stop)
            reason = data.get("done_reason")
            if not strip_thinking(content):
                thinking = str(msg.get("thinking") or "")
                if data.get("aborted"):
                    raise NoAnswer(f"{data['aborted']} and was stopped ({len(thinking)} thinking chars, no answer)")
                raise NoAnswer(
                    f"the model sent no answer (think={think!r}, done_reason={reason!r}, "
                    f"{data.get('eval_count', '?')} tokens, num_predict={options['num_predict']}, "
                    f"answer chars={len(content)}, thinking chars={len(thinking)}"
                    + (f", thinking ends: {thinking[-160:]!r}" if thinking else "")
                    + (", content starts: " + repr(content[:160]) if content.strip() else "") + "). "
                    + ("It ran out of tokens while thinking: raise CODEAGENT_NUM_PREDICT or lower CODEAGENT_THINK_LEVEL."
                       if reason == "length" else
                       "It stopped on its own before answering: this model may not handle think=true "
                       "(try CODEAGENT_THINK=false, or check `ollama show <model>` for thinking support)."))
            return self._finish(content, reason == "length")
        except (KeyError, TypeError, AttributeError) as e:
            raise LLMError("unexpected response shape from Ollama") from e


class OpenAIClient(_BaseClient):
    def chat(self, messages: list[dict], *, temperature: float | None = None,
             num_predict: int | None = None, stop: list[str] | None = None,
             on_token: Callable[[str, str], None] | None = None,       # on_token: not streamed on this backend
             think: bool | None = None) -> Reply:
        c = self.cfg
        payload = {
            "model": c.model, "messages": messages, "stream": False,
            "max_tokens": num_predict if num_predict is not None else c.num_predict,
            "temperature": temperature if temperature is not None else c.temperature,
        }
        if stop:
            payload["stop"] = stop
        payload["chat_template_kwargs"] = {"enable_thinking": c.think and think is not False}   # llama.cpp; other servers ignore it
        data = self._request("/v1/chat/completions", payload)
        try:
            choice = data["choices"][0]
            return self._finish(choice["message"]["content"], choice.get("finish_reason") == "length")
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError("unexpected response shape from the server") from e


def make_client(cfg: Config, **kw) -> LLMClient:
    cls = OllamaClient if cfg.backend == "ollama" else OpenAIClient
    return cls(cfg, **kw)


def list_models(cfg: Config) -> list[str]:
    client = OllamaClient(cfg, retries=0) if cfg.backend == "ollama" else OpenAIClient(cfg, retries=0)
    if cfg.backend == "ollama":
        data = client._request("/api/tags", None)
        return [m["name"] for m in data.get("models", [])]
    data = client._request("/v1/models", None)
    return [m["id"] for m in data.get("data", [])]


# ==========================================================================
# prompts
# Prompts for a very small model: short, rigid, one job at a time.
# 
# The harness (not the model) decides what happens next; the model only ever
# answers "write these specific files", in one fixed output format.
# ==========================================================================

SYSTEM = """You are a careful programmer. You write small, correct, working code.

Reply with ONLY file blocks, in exactly this format, and nothing else:

FILE: path/to/file.py
```python
<the COMPLETE content of the file>
```

Rules:
- Always write the COMPLETE file, never a diff or a snippet.
- Write only the files you are asked to write, with exactly those paths.
- No explanations, no text outside the file blocks.
- No comments and no docstrings: write only the code, as short as possible.
- Do exactly what the description says: no extra validation, no type checks, no extra features.
- An empty file is a block with nothing inside the fence."""


_E_NOISE = ("assert ", "At index", "Use -v", "Right contains", "Left contains", "Full diff", "Differing",
            "Omitting", "Extra items", "Drill down", "Full output")


def condense_test_output(output: str, limit: int = 1500, max_failures: int = 8) -> str:
    """What the model needs from a pytest run: one line per failing test (its own message, which already says
    what was returned and what was expected) and the summary line; none of pytest's repeated diff noise."""
    lines = [ln for ln in output.splitlines() if not ln.startswith("...[output truncated")]
    head = re.compile(r"^_{2,} (.+?) _{2,}$")
    found: list[str] = []
    test, seen = "", set()
    for line in lines:
        m = head.match(line)
        if m:
            test = m.group(1)
            continue
        if not (line.startswith("E ") or line == "E"):
            continue
        msg = line[1:].strip()
        if not msg or msg.startswith(_E_NOISE) or msg[0] in "+-?" or (test, "x") in seen:
            continue
        seen.add((test, "x"))
        found.append(f"{test + ': ' if test else ''}{msg[:500]}")
    summary = next((ln.strip() for ln in reversed(lines) if re.search(r"\d+ (failed|passed|error)", ln)), "")
    if not found:
        text = "\n".join(lines).strip()
        if len(text) > limit:
            text = text[-limit:]
            text = text[text.find("\n") + 1:] if "\n" in text else text
        return text
    if len(found) > max_failures:
        found = found[:max_failures] + [f"... and {len(found) - max_failures} more failing tests"]
    out = "\n".join(found + ([summary] if summary else []))
    return out if len(out) <= limit else out[:limit].rsplit("\n", 1)[0]


def function_body(source: str, name: str) -> str | None:
    """The lines inside `def name(...)`, dedented, or None if there is no such function."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            lines = source.splitlines()[node.body[0].lineno - 1: node.end_lineno]
            return textwrap.dedent("\n".join(lines)).strip("\n")
    return None


def _fit(files: dict[str, str | None], budget: int) -> str:
    """Render files as FILE blocks, sharing `budget` characters between them."""
    present = {p: c for p, c in files.items() if c is not None}
    if not present:
        return "(none)"
    share = max(200, budget // len(present))
    out = []
    for path, content in present.items():
        if len(content) > share:
            content = content[:share].rstrip() + "\n# ... (truncated)\n"
        out.append(f"FILE: {path}\n```\n{content}{'' if content.endswith(chr(10)) or not content else chr(10)}```")
    return "\n\n".join(out)


SYSTEM_BODY = """You are a careful programmer. You write small, correct, working code.

You are given a function signature. Reply with ONLY the lines that go inside the function, in exactly this format, and nothing else:

FILE: path/to/file.py
```python
<body lines only>
```

Rules:
- Write only the body: no def line, no docstring, no comments, no explanation.
- Imports the body needs go first, before the body lines.
- Do exactly what the description says: no extra validation, no type checks, no extra features.
- Keep it short. Do not think for long: decide, then write the body."""


def _example_line(name: str, ex: Example) -> str:
    call = _call(name, ex)
    return f"- {call} raises {ex.raises}" if ex.raises else f"- {call} -> {ex.expect!r}"


def _name_the_calls(report: str, name: str, examples: tuple[Example, ...]) -> str:
    """`test_example_2_raises_ValueError: Failed: DID NOT RAISE ValueError` says nothing about WHICH call;
    the harness knows, so it says it."""
    def line(m: re.Match) -> str:
        i = int(m.group(1))
        if not 1 <= i <= len(examples):
            return m.group(0)
        call, rest = _call(name, examples[i - 1]), m.group(2)
        m2 = re.match(r"Failed: DID NOT RAISE (\w+)", rest)
        return f"{call} must raise {m2.group(1)}, but it returned normally" if m2 else f"{call}: {rest}"
    return re.sub(r"^test_example_(\d+)(?:_raises_\w+)?: (.*)$", line, report, flags=re.M)


def build_body_prompt(plan: Plan, step: Step, tests: dict[str, str], existing: dict[str, str | None],
                      budget: int, test_output: str | None, feedback: str | None) -> list[dict]:
    """Skeleton mode: the model sees the task, the def line, the examples, how its last body failed, and
    what the tests said. Never the pytest file, the package __init__, or a whole module with a def line."""
    path, name = module_path(step), _function_name(step)
    parts = [_header(plan, step),
             f"The function line is already written. Write ONLY its body:\n\ndef {step.signature}:\n    ..."]
    if step.examples:
        parts.append("The tests check exactly these examples:\n" + "\n".join(_example_line(name, e)[:300] for e in step.examples))
    else:
        parts.append("These tests must pass:\n" + _fit(dict(tests), budget // 2))
    ctx = {p: existing.get(p) for p in step_context(step)}
    if any(v is not None for v in ctx.values()):
        parts.append("Related files (read-only):\n" + _fit(ctx, budget // 2))
    ready = [ln.rsplit(" import ", 1)[1] for ln in context_imports(step, existing)]
    if ready:
        parts.append("Already imported for you: " + ", ".join(ready) + ". Do not write imports for them.")
    prev = function_body(existing.get(path) or "", name)
    if prev and prev.strip() != "pass":
        parts.append("Your previous body, which failed (do not repeat it):\n```python\n" + prev[:budget // 2] + "\n```")
    if re.search(r"^\s*\d+[.)]\s", step.description, re.M):
        parts.append("Follow the numbered steps above in order, one after the other: do not skip a step "
                     "and do not add checks of your own.")
    if test_output:
        parts.append("What the tests reported:\n" + _name_the_calls(condense_test_output(test_output), name, step.examples))
    if feedback:
        parts.append("Problem with your previous answer, fix it:\n" + feedback)
    parts.append("Reply with ONLY this, the body lines and nothing else (no def line, no docstring, "
                 f"no comments, no explanation):\nFILE: {path}\n```python\n<body lines>\n```")
    return [{"role": "system", "content": SYSTEM_BODY}, {"role": "user", "content": "\n\n".join(parts)}]


def _header(plan: Plan, step: Step) -> str:
    return f"Project: {plan.title}\nTask: {step.title}\n\n{step.description.strip()}\n"


def build_tests_prompt(plan: Plan, step: Step, existing: dict[str, str | None], budget: int,
                   feedback: str | None) -> list[dict]:
    parts = [
        _header(plan, step),
        "Write the TEST file(s) first (test-driven development).",
        "The tests must FAIL until the feature is implemented, so test the real behaviour with "
        "concrete examples and assert exact results.",
        "Every test is a plain function with NO parameters and the input written inside it, like: "
        "def test_lower():\n    assert f('A') == 'a'. Write 3 to 6 short tests, no fixtures, no loops.",
        f"Test file(s) to write: {', '.join(step.tests)}",
        f"The code under test will be written later in: {', '.join(step.files)}",
    ]
    ref = {p: existing.get(p) for p in (*step.files, *step.context)}
    if any(v is not None for v in ref.values()):
        parts.append("Existing files for reference (read-only):\n" + _fit(ref, budget))
    if feedback:
        parts.append("Problem with your previous answer, fix it:\n" + feedback)
    parts.append(f"Reply with ONLY the file block(s) for: {', '.join(step.tests)}. "
                 f"Do NOT write {', '.join(step.files)} now: the implementation comes later.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def build_impl_prompt(plan: Plan, step: Step, tests: dict[str, str], existing: dict[str, str | None],
                  budget: int, test_output: str | None, feedback: str | None) -> list[dict]:
    if step.signature:
        return build_body_prompt(plan, step, tests, existing, budget, test_output, feedback)
    parts = [
        _header(plan, step),
        "These tests must pass. Do NOT change the tests:\n" + _fit(dict(tests), budget // 2),
        f"Write the implementation file(s): {', '.join(step.files)}",
    ]
    if step.function:
        module, name = step.function.split(":")
        parts.append(f"The tests call `{name}` imported from {module}: define exactly `def {name}(...)` "
                     f"at the top level of {module.replace('.', '/')}.py.")
    cur = {p: existing.get(p) for p in (*step.files, *step.context)}
    if any(v is not None for v in cur.values()):
        parts.append("Current content of the files:\n" + _fit(cur, budget // 2))
    if test_output:
        parts.append("Result of running the tests (fix what fails):\n" + condense_test_output(test_output))
    if feedback:
        parts.append("Problem with your previous answer, fix it:\n" + feedback)
    parts.append(f"Reply with ONLY the file block(s) for: {', '.join(step.files)}. "
                 f"Do NOT write or change {', '.join(step.tests)}: the tests are already final.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


# ==========================================================================
# runner
# The agent loop. The HARNESS drives; the model only answers narrow questions.
# 
# For every plan step (test-driven, like a careful human pair):
#   1. ask the model for the TEST files; they must fail before any code exists
#   2. ask for the IMPLEMENTATION files; run the tests; on failure feed the real
#      output back and retry (bounded)
#   3. commit exactly the step's files, record progress
# Anything that goes wrong rolls the step's files back to how they were.
# ==========================================================================

class LivePrinter:
    """The prompt sent to the model and its raw output as it is generated, in the terminal only.
    Nothing here is written to the log file or the replies folder."""

    def __init__(self, write: Callable[[str], None] | None = None, color: bool | None = None):
        self.write = write or (lambda text: (sys.stdout.write(text), sys.stdout.flush()))
        self.color = sys.stdout.isatty() if color is None else color
        self.kind: str | None = None
        self.streamed = False

    def _paint(self, text: str, code: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.color else text

    def prompt(self, messages: list[dict], label: str) -> None:
        self.kind, self.streamed = None, False
        self.write("\n" + self._paint(f"==== PROMPT: {label} ====", "1;36") + "\n")
        for m in messages:
            self.write(self._paint(f"[{m['role']}]", "36") + "\n" + str(m["content"]).rstrip() + "\n")

    def chunk(self, kind: str, text: str) -> None:
        self.streamed = True
        if kind != self.kind:
            self.kind = kind
            title, code = ("MODEL THINKING", "1;33") if kind == "thinking" else ("MODEL ANSWER (raw)", "1;32")
            self.write("\n" + self._paint(f"==== {title} ====", code) + "\n")
        self.write(self._paint(text, "2") if kind == "thinking" else text)

    def end(self) -> None:
        self.write(("\n" if self.kind else "") + self._paint("==== END ====", "1;35") + "\n")
        self.kind = None


class Transcript:
    """Every prompt sent to the model and every reply, in one readable file (.codeagent/transcript.md).
    Thinking is included when the reply was streamed (run --live). Writing here must never crash the agent."""

    def __init__(self, root: str | Path):
        self.path = Path(root) / ".codeagent" / "transcript.md"
        self._thinking: list[str] = []
        self._content: list[str] = []

    def _append(self, text: str) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(text)
        except OSError:
            pass

    def prompt(self, messages: list[dict], label: str) -> None:
        self._thinking, self._content = [], []
        body = "".join(f"[{m['role']}]\n{str(m['content']).rstrip()}\n\n" for m in messages)
        self._append(f"\n==== PROMPT: {label} ({time.strftime('%Y-%m-%d %H:%M:%S')}) ====\n{body}")

    def chunk(self, kind: str, text: str) -> None:
        (self._thinking if kind == "thinking" else self._content).append(text)

    def no_reply(self, reason: str, *, seconds: float) -> None:
        """The call ended without a usable answer: keep whatever streamed in, and why it ended."""
        out = ""
        if self._thinking:
            out += f"==== MODEL THINKING (partial) ====\n{''.join(self._thinking).rstrip()}\n\n"
        if self._content:
            out += f"==== MODEL ANSWER (partial) ====\n{''.join(self._content).rstrip()}\n\n"
        self._thinking, self._content = [], []
        self._append(out + f"==== NO USABLE ANSWER after {seconds:.0f}s: {reason} ====\n")

    def reply(self, text: str, *, seconds: float, truncated: bool, temperature: float) -> None:
        out = ""
        if self._thinking:
            out += f"==== MODEL THINKING ====\n{''.join(self._thinking).rstrip()}\n\n"
        out += (f"==== MODEL ANSWER (raw) ====\n# seconds: {seconds:.1f}\n# truncated: {'yes' if truncated else 'no'}\n"
                f"# temperature: {temperature:g}\n{text.rstrip()}\n==== END ====\n")
        self._thinking, self._content = [], []
        self._append(out)


class _Pause(Exception):
    """Not the model's fault (server down, low resources, stop requested): retry later."""


class _Fail(Exception):
    """The step could not be completed."""


class _TestsBroken(Exception):
    """The tests themselves cannot run (not an implementation problem): rewrite them."""


@dataclass
class StepResult:
    title: str
    status: str                       # done | failed | skipped | paused
    attempts: int = 0
    reason: str = ""
    commit: str | None = None


@dataclass
class RunReport:
    plan_title: str
    branch: str
    results: list[StepResult] = field(default_factory=list)

    @property
    def outcome(self) -> str:
        statuses = {r.status for r in self.results}
        if "failed" in statuses:
            return "failed"
        if "paused" in statuses:
            return "paused"
        return "done"

    def summary(self) -> str:
        lines = [f"Plan: {self.plan_title}  (branch {self.branch})  ->  {self.outcome.upper()}"]
        for r in self.results:
            extra = f"commit {r.commit}, {r.attempts} model calls" if r.status == "done" else r.reason
            lines.append(f"  [{r.status}] {r.title}" + (f"  ({extra})" if extra else ""))
        return "\n".join(lines)


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items)


class Runner:
    def __init__(self, cfg: Config, plan: Plan, ws: Workspace, llm: LLMClient, *,
                 guard=None, run_cmd=run_command, log: Callable[[str], None] | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 should_stop: Callable[[], bool] = lambda: False, live: LivePrinter | None = None):
        first = shlex.split(plan.test_command)[0]
        if first not in cfg.allowed_commands:
            raise PlanError(f"test_command {first!r} is not in allowed_commands {cfg.allowed_commands}")
        self.cfg, self.plan, self.ws, self.llm = cfg, plan, ws, llm
        self.guard = guard or ResourceGuard(cfg, ws.root)
        self.run_cmd = run_cmd
        self.log = log or (lambda m: None)
        self.sleep = sleep
        self.should_stop = should_stop
        self.live = live
        self.attempts = 0
        self.replies = ReplyLog(ws.root)
        self.transcript = Transcript(ws.root)

    # ------------------------------------------------------------------ run
    def run(self) -> RunReport:
        with WorkspaceLock(self.ws.root):
            ensure_repo(self.ws.root)
            branch = f"agent/{plan_slug(self.plan)}"
            checkout_branch(self.ws.root, branch)
            state = self.state = RunState(self.ws.root)
            report = RunReport(self.plan.title, branch)
            for step in self.plan.steps:
                if self.should_stop():
                    report.results.append(StepResult(step.title, "paused", reason="stop requested"))
                    break
                if state.is_done(self.plan, step):
                    gone = [f for f in (*step.files, *step.tests) if not (self._read_many([f])[f] or "").strip()
                            and not f.endswith("__init__.py")]
                    if not gone:
                        report.results.append(StepResult(step.title, "skipped"))
                        continue
                    self.log(f"== step: {step.title}: marked done, but {', '.join(gone)} missing or empty; running it again")
                self.log(f"== step: {step.title}")
                result = self._run_step(step)
                self.log(f"   -> {result.status} {result.reason}".rstrip())
                report.results.append(result)
                if result.status == "paused":
                    break
                if result.status == "failed" and self.cfg.stop_on_failure:
                    break
            return report

    # ------------------------------------------------------------------ one step
    def _run_step(self, step: Step) -> StepResult:
        paths = (*step.tests, *step.files)
        snapshot = self.ws.snapshot(paths)
        self.attempts = 0
        try:
            feedback = None
            for rewrite in range(self.cfg.max_test_rewrites + 1):
                tests, satisfied, output = self._tests_phase(step, feedback)
                if satisfied:
                    self.log("   implementation already satisfies the tests")
                    break
                try:
                    self._impl_phase(step, tests, output)
                    break
                except _TestsBroken as broken:
                    if step.examples:                       # generated from the plan: rewriting cannot help
                        raise _Fail(f"the tests generated from the plan's examples cannot run: {broken}") from broken
                    if rewrite == self.cfg.max_test_rewrites:
                        raise _Fail(f"the tests are broken and rewriting them did not help: {broken}") from broken
                    self.log(f"   the tests cannot run ({broken}); rewriting the tests")
                    self.ws.restore(snapshot)               # drop the half-built implementation too
                    feedback = str(broken)
            self._write(tests)                             # tests are exactly what we approved
            sha = commit_paths(self.ws.root, paths, step.commit)
            self.state.mark_done(self.plan, step, sha)
            return StepResult(step.title, "done", self.attempts, commit=sha)
        except _Pause as p:
            self.ws.restore(snapshot)
            return StepResult(step.title, "paused", self.attempts, reason=str(p))
        except (_Fail, GitError, WorkspaceError) as e:
            self.ws.restore(snapshot)
            return StepResult(step.title, "failed", self.attempts, reason=str(e))

    def _spec_tests_phase(self, step: Step):
        """The plan's examples ARE the tests: printed by the harness, never written by the model."""
        files = {step.tests[0]: render_tests(step)}
        self.log(f"   tests generated from {len(step.examples)} example(s) in the plan (no model call)")
        self._write(files)
        res = self._run_tests()
        if res.timed_out or res.flooded:
            raise _Fail("the tests generated from the plan's examples hung or printed endlessly; check the examples")
        if res.ok:
            if all(self.ws.exists(f) for f in step.files):
                return files, True, res.output
            raise _Fail("the plan's examples already pass with no implementation; check the examples")
        return files, False, res.output

    def _tests_phase(self, step: Step, feedback: str | None = None):
        if step.examples:
            return self._spec_tests_phase(step)
        for n in range(1, self.cfg.max_attempts + 1):
            existing = self._read_many((*step.files, *step.context))
            text = self._ask(build_tests_prompt(self.plan, step, existing, self.cfg.max_context_chars, feedback),
                             step=step, phase="tests", attempt=n,
                             num_predict=min(self.cfg.num_predict_tests, self.cfg.num_predict))
            files, problems = self._collect(text, step.tests)
            problems += self._lint_tests(files)
            if problems:
                feedback = _bullets(problems)
                self.log(f"   tests attempt {n}: {problems[0]}")
                continue
            self._write(files)
            res = self._run_tests()
            if res.timed_out or res.flooded:
                feedback = "Your tests hung or printed endlessly. Write simple, fast tests with no loops or input()."
                continue
            if res.ok:
                if all(self.ws.exists(f) for f in step.files):
                    return files, True, res.output
                feedback = ("These tests PASS even though the feature is not implemented yet, so they test "
                            "nothing. Assert concrete results from the description so they FAIL first.")
                self.log(f"   tests attempt {n}: vacuous (passed with no implementation)")
                continue
            broken = diagnose_broken_tests(res.output)
            if broken:                                      # red for the wrong reason: the tests are the problem
                feedback = broken
                self.log(f"   tests attempt {n}: tests cannot run ({broken})")
                continue
            return files, False, res.output                 # red: exactly what we want
        raise _Fail(f"could not get valid failing tests after {self.cfg.max_attempts} attempts: "
                    f"{(feedback or '').splitlines()[0] if feedback else 'unknown'}")

    def _impl_phase(self, step: Step, tests: dict[str, str], output: str) -> None:
        feedback = None
        seen: dict[str, int] = {}                        # code we already tried -> attempt number
        fresh = False                                    # next prompt hides the previous code (anti-anchoring)
        repeats = 0                                      # how many times the model repeated itself
        quiet = bool(step.signature) and self.cfg.think and not self.cfg.think_body   # a body is written without thinking
        if step.signature:
            gone = missing_named_files(step, self._read_many(step_context(step)))
            if gone:
                raise _Fail(f"the step description names {', '.join(gone)}, but it is missing or empty in the "
                            f"workspace; an earlier step must create it first (check that it passed and was committed)")
            output, satisfied = self._reset_skeleton(step, tests, output)
            if satisfied:
                return
        for n in range(1, self.cfg.max_attempts + 1):
            existing = self._read_many((*step.files, *step_context(step)))
            if fresh:                                    # hide only the model's own old code, never the related files
                existing.update({p: None for p in step.files})

            text = self._ask(build_impl_prompt(self.plan, step, tests, existing,
                                               self.cfg.max_context_chars, output, feedback),
                             step=step, phase="impl", attempt=n,
                             num_predict=self.cfg.num_predict_body if step.signature else self.cfg.num_predict,
                             stop=BODY_STOP if step.signature else None, bump=0.15 * repeats,
                             think=False if quiet else None)
            fresh = False
            if getattr(text, "no_answer", ""):
                quiet = True
                repeats += 1
            if step.signature:                       # skeleton mode: the model wrote only the body
                files, problems = self._collect(text, (module_path(step),), check_syntax=False)  # a body is not a module
                if not problems:
                    inits = {f: self._read_many([f])[f] for f in step.files if f.endswith("__init__.py")}
                    files, problems = assemble_files(step, files, inits, self._read_many(step_context(step)))
            else:
                files, problems = self._collect(text, step.files)
            if not problems:
                problems += self._lint_impl(step, files)
            if not problems and self.cfg.tidy_code:
                files = {p: tidy_python(c) if p.endswith(".py") else c for p, c in files.items()}
            if problems:
                feedback = _bullets(problems)
                self.log(f"   impl attempt {n}: {problems[0]}")
                continue
            key = "\n".join(f"{p}\n{(function_body(files[p], _function_name(step)) if step.signature and p == module_path(step) else None) or files[p]}"
                            for p in sorted(files))
            if key in seen:                              # a tiny model often copies its last answer verbatim
                repeats += 1
                fresh = True
                feedback = (f"this is identical to your attempt {seen[key]}, which already failed the tests. "
                            f"Write a DIFFERENT solution; do not copy your earlier code.")
                self.log(f"   impl attempt {n}: same code as attempt {seen[key]}; not re-running, asking for a different one")
                continue
            seen[key] = n
            self._write(files)
            self._write(tests)
            problems = self._static_check(list(files))
            if problems:
                feedback = _bullets(problems)
                self.log(f"   impl attempt {n}: static check: {problems[0]}")
                continue
            feedback = None
            res = self._run_tests()
            if res.ok:
                return
            broken = diagnose_broken_tests(res.output)
            if broken:
                raise _TestsBroken(broken)
            output = res.output
            self.log(f"   impl attempt {n}: tests still failing")
        tail = output.strip().splitlines()[-1] if output.strip() else ""
        raise _Fail(f"tests still failing after {self.cfg.max_attempts} attempts. Last line: {tail}")

    # ------------------------------------------------------------------ pieces
    def _reset_skeleton(self, step: Step, tests: dict[str, str], output: str) -> tuple[str, bool]:
        """Start every skeleton step from a clean `pass` body, so leftovers of a killed or failed run never
        mislead the model and the test output it sees describes exactly the code it is asked to replace."""
        inits = {f: self._read_many([f])[f] for f in step.files if f.endswith("__init__.py")}
        files, problems = assemble_files(step, {module_path(step): "pass"}, inits)
        if problems:
            return output, False
        self._write(files)
        self._write(tests)
        res = self._run_tests()
        if res.timed_out or res.flooded:
            return output, False
        return res.output, res.ok

    def _retry_temperature(self, attempt: int, bump: float = 0.0, base: float | None = None) -> float:
        """Attempt 1 uses the configured temperature; retries warm up only a little, and never past
        the cap. A tiny model gets much worse when hot (it starts writing nonsense)."""
        base = self.cfg.temperature if base is None else base
        ceiling = max(base, self.cfg.max_retry_temperature)
        t = min(base + self.cfg.retry_temperature_step * (attempt - 1), ceiling)
        return min(t + bump, max(ceiling, 0.7)) if bump else t     # repeating itself earns a warmer retry

    def _static_check(self, paths: list[str]) -> list[str]:
        """ruff, if installed: remove unused imports, then report undefined names before running anything."""
        if not self.cfg.use_ruff:
            return []
        ruff_fix(self.ws.root, paths)
        return ruff_problems(self.ws.root, paths)

    def _lint_impl(self, step: Step, files: dict[str, str]) -> list[str]:
        """A tool, not a prompt: the function the examples call must really be defined."""
        if not step.function:
            return []
        module, name = step.function.split(":")
        path = module.replace(".", "/") + ".py"
        if path not in files:
            return []
        if name not in defined_names(files[path]):
            return [f"{path} does not define `{name}`; write `def {name}(...)` at the top level of {path}"]
        if any(not e.raises for e in step.examples) and returns_a_value(files[path], name) is False:
            return [f"`{name}` never returns a value, but the examples expect it to return a result; "
                    f"end it with `return <the result>`"]
        return []

    def _lint_tests(self, files: dict[str, str]) -> list[str]:
        fixtures = set()
        try:
            conftest = self.ws.read("tests/conftest.py") or ""
            fixtures = {n.name for n in ast.walk(ast.parse(conftest)) if isinstance(n, ast.FunctionDef)}
        except (WorkspaceError, SyntaxError):
            pass
        out = []
        for path, content in files.items():
            if path.endswith(".py"):
                out += [f"{path}: {p}" for p in check_test_source(content, fixtures)]
        return out

    def _ask(self, messages: list[dict], *, step: Step, phase: str, attempt: int,
             num_predict: int | None = None, stop: list[str] | None = None, bump: float = 0.0,
             think: bool | None = None) -> str:
        if self.should_stop():
            raise _Pause("stop requested")
        ok = self.guard.wait_until_ok(on_wait=lambda r: self.log("   waiting for resources: " + "; ".join(r)),
                                      should_stop=self.should_stop)
        if not ok:
            if self.should_stop():
                raise _Pause("stop requested")
            raise _Pause("system resources stayed too low; will retry later")
        temperature = self._retry_temperature(attempt, bump, self.cfg.temperature_no_think
                                              if (think is False and self.cfg.think) else None)
        started = time.monotonic()
        self.log(f"   asking the model ({phase}, attempt {attempt}, think={(False if think is False else think_value(self.cfg))!r}); "
                 "a reasoning model can be silent for minutes")
        try:
            self.attempts += 1
            extra = {"stop": stop} if stop else {}
            if think is False:
                extra["think"] = False
            label = f"{step.title} / {phase} / attempt {attempt}" + (" / no thinking" if think is False else "")
            self.transcript.prompt(messages, label)
            if self.live:
                self.live.prompt(messages, label)

            def on_token(kind: str, piece: str) -> None:        # always streamed: a loop can be stopped, a partial kept
                self.transcript.chunk(kind, piece)
                if self.live:
                    self.live.chunk(kind, piece)
            text = self.llm.chat(messages, temperature=temperature, num_predict=num_predict,
                                 on_token=on_token, **extra)
            if self.live and not self.live.streamed:        # a backend that does not stream: show the whole reply now
                self.live.chunk("content", str(text))
        except NoAnswer as e:                                # looped or ran out of tokens thinking: a failed attempt
            seconds = time.monotonic() - started
            self.transcript.no_reply(str(e), seconds=seconds)
            self.log(f"   no usable answer after {seconds:.0f}s: {e}")
            text = _reply("", True)
            text.no_answer = _no_answer_feedback(e)
            self.sleep(self.cfg.step_cooldown_s)
            return text
        except LLMError as e:
            self.transcript.no_reply(f"model error: {e}", seconds=time.monotonic() - started)
            raise _Pause(f"model unavailable: {e}") from e
        except BaseException as e:                           # Ctrl+C and friends: still keep what streamed in
            self.transcript.no_reply(f"interrupted ({type(e).__name__})", seconds=time.monotonic() - started)
            raise
        finally:
            if self.live:
                self.live.end()
        seconds = time.monotonic() - started
        truncated = bool(getattr(text, "truncated", False))
        self.replies.save(step.title, phase, attempt, str(text), seconds=seconds, truncated=truncated,
                          temperature=temperature)
        self.transcript.reply(str(text), seconds=seconds, truncated=truncated, temperature=temperature)
        self.log(f"   model replied in {seconds:.0f}s ({len(text)} chars)"
                 + (" - CUT OFF at the token limit" if truncated else ""))
        self.sleep(self.cfg.step_cooldown_s)              # let the laptop breathe
        return text

    def _collect(self, text: str, allowed: tuple[str, ...], *,
                 check_syntax: bool = True) -> tuple[dict[str, str], list[str]]:
        if getattr(text, "no_answer", ""):
            return {}, [text.no_answer]
        parsed = parse_file_blocks(str(text), truncated=bool(getattr(text, "truncated", False)))
        for note in parsed.notes:
            self.log(f"   fixed up: {note}")
        problems = list(parsed.problems)
        targets = [p for p in allowed if not p.endswith("__init__.py")]
        if not parsed.blocks and len(parsed.anonymous) == 1 and len(targets) == 1:
            parsed.blocks = [FileBlock(targets[0], parsed.anonymous[0])]     # only one file is possible: no header needed
            self.log(f"   fixed up: the reply had no FILE header; used its only code block as {targets[0]}")
        if not parsed.blocks and not problems:
            return {}, ["your reply contained no FILE blocks; reply only with: FILE: <path> then a fenced code block"]
        got: dict[str, str] = {}
        ignored = []
        for b in parsed.blocks:
            (got.__setitem__(b.path, b.content) if b.path in allowed else ignored.append(b.path))
        if ignored:
            self.log(f"   ignored files outside this step: {', '.join(ignored)}")
        for path in allowed:
            if path in got:
                continue
            if path.endswith("__init__.py"):
                if not self.ws.exists(path):
                    got[path] = ""                          # trivial file: create it for the model
                continue
            problems.append(f"missing file: {path} (write every requested file, complete)")
        if ignored and problems:
            problems.insert(0, f"you wrote {', '.join(ignored)}, which this step does not allow. "
                               f"Write ONLY: {', '.join(allowed)}")
        for path, content in got.items():
            if len(content.encode("utf-8")) > self.cfg.max_file_bytes:
                problems.append(f"{path}: file is too large")
            elif check_syntax and path.endswith(".py"):
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    problems.append(f"{path}: syntax error on line {e.lineno}: {e.msg}")
        return got, problems

    def _write(self, files: dict[str, str]) -> None:
        for path, content in files.items():
            self.ws.write(path, content)

    def _read_many(self, paths) -> dict[str, str | None]:
        out = {}
        for p in paths:
            try:
                out[p] = self.ws.read(p)
            except WorkspaceError:
                out[p] = None
        return out

    def _run_tests(self) -> CommandResult:
        return self.run_cmd(self.plan.test_command, self.ws.root, allowed=self.cfg.allowed_commands,
                            timeout=self.cfg.test_timeout, mem_mb=self.cfg.command_mem_mb,
                            max_output_chars=max(self.cfg.max_output_chars * 5, 15000))


# ==========================================================================
# cli
# Command line: template | check | doctor | smoke | run
# ==========================================================================

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_PAUSED, EXIT_BUSY = 0, 1, 2, 3, 4


class _Stopper:
    """First Ctrl+C/SIGTERM = finish the current call and stop cleanly; second = abort now."""

    def __init__(self):
        self.flag = False
        self.count = 0
        self._old: dict = {}

    def __call__(self) -> bool:
        return self.flag

    def _handle(self, signum, frame):
        self.count += 1
        self.flag = True
        if self.count >= 2:
            raise KeyboardInterrupt

    def install(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._old[sig] = signal.signal(sig, self._handle)

    def restore(self):
        for sig, old in self._old.items():
            signal.signal(sig, old)
        self._old.clear()


def _make_log(out, log_file: Path | None):
    """Print with a timestamp and also append to a size-capped rotating log file."""
    logger = None
    handler = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3,
                                                       encoding="utf-8")
        logger = logging.getLogger(f"codeagent.{id(handler)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.addHandler(handler)

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        out(line)
        if logger:
            logger.info(line)

    def close():
        if handler:
            logger.removeHandler(handler)
            handler.close()

    return log, close


def _check_test_command(cfg: Config, plan) -> None:
    first = shlex.split(plan.test_command)[0]
    if first not in cfg.allowed_commands:
        raise PlanError(f"test_command {first!r} is not in allowed_commands {cfg.allowed_commands}")


# ------------------------------------------------------------------ commands

def cmd_template(args, out, **_):
    if args.out:
        dest = Path(args.out)
        if dest.exists() and not args.force:
            out(f"{dest} already exists (use --force to overwrite)")
            return EXIT_USAGE
        dest.write_text(TEMPLATE, encoding="utf-8")
        out(f"wrote {dest}")
    else:
        out(TEMPLATE)
    return EXIT_OK


def cmd_check(args, out, **_):
    cfg = load_config()
    plan = load_plan(args.plan)
    _check_test_command(cfg, plan)
    out(f"Plan OK: {plan.title}  ({len(plan.steps)} step{'s' if len(plan.steps) != 1 else ''})")
    out(f"  test command: {plan.test_command}")
    for i, s in enumerate(plan.steps, 1):
        out(f"  {i}. {s.title}")
        out(f"       tests: {', '.join(s.tests)}")
        out(f"       files: {', '.join(s.files)}")
        out(f"       commit: {s.commit}")
    return EXIT_OK


def cmd_doctor(args, out, **_):
    cfg = load_config()
    failed = False

    def ok(m): out(f"[ok]   {m}")
    def warn(m): out(f"[warn] {m}")
    def fail(m):
        nonlocal failed
        failed = True
        out(f"[FAIL] {m}")

    if sys.version_info >= (3, 11):
        ok(f"python {sys.version.split()[0]}")
    else:
        fail(f"python {sys.version.split()[0]} is too old (need 3.11+)")
    ok("git found") if shutil.which("git") else fail("git is not installed (sudo apt install git)")

    out(f"       backend={cfg.backend} url={cfg.base_url} model={cfg.model} num_ctx={cfg.num_ctx}")
    if cfg.think and cfg.think_level != "low" and not cfg.model.lower().startswith("gpt-oss"):
        warn(f"CODEAGENT_THINK_LEVEL={cfg.think_level} does nothing for {cfg.model}: only gpt-oss models take a "
             "level, every other model just thinks on/off. Use CODEAGENT_MAX_THINK_CHARS to bound the thinking.")
    if cfg.think and cfg.temperature < 0.5:
        warn("thinking models loop more often at low temperature; 0.6 is the usual setting for thinking mode.")
    try:
        models = list_models(cfg)
    except LLMError as e:
        fail(str(e))
    else:
        wanted = {cfg.model, f"{cfg.model}:latest"}
        if wanted & set(models):
            ok(f"model {cfg.model} is installed")
        else:
            avail = ", ".join(models) or "(none)"
            fail(f"model {cfg.model!r} is not installed. Installed: {avail}. "
                 "Set CODEAGENT_MODEL.")

    guard = ResourceGuard(cfg, Path.cwd())
    mem, bat, temp, disk = guard.free_memory_mb(), guard.battery(), guard.max_temp_c(), guard.free_disk_mb()
    out(f"       memory free: {mem if mem is not None else 'n/a'} MB | "
        f"battery: {f'{bat[0]}% ' + ('(discharging)' if bat[1] else '(plugged in)') if bat else 'none'} | "
        f"cpu temp: {f'{temp:.0f}C' if temp is not None else 'n/a'} | "
        f"disk free: {disk if disk is not None else 'n/a'} MB")
    problems = guard.problems()
    for p in problems:
        warn(p + "  (the agent would pause right now)")
    if not problems:
        ok("resources look fine")
    return EXIT_FAILED if failed else EXIT_OK


def cmd_smoke(args, out, **_):
    """Can this model follow the FILE-block output format at all? (needs the model server)"""
    cfg = load_config()
    client = make_client(cfg)
    prompt = [{"role": "system", "content": SYSTEM},
              {"role": "user", "content": "Write the file math_utils.py containing a function "
                                          "add(a, b) that returns the sum of a and b."}]
    good = 0
    for i in range(1, args.tries + 1):
        try:
            reply = client.chat(prompt)
            blocks = {b.path: b.content for b in parse_file_blocks(reply).blocks}
            tree = ast.parse(blocks["math_utils.py"])
            assert any(isinstance(n, ast.FunctionDef) and n.name == "add" for n in ast.walk(tree))
            good += 1
            out(f"  try {i}: ok")
        except LLMError as e:
            out(f"  try {i}: model error: {e}")
        except (KeyError, SyntaxError, AssertionError):
            out(f"  try {i}: reply did not follow the format")
    out(f"format compliance: {good}/{args.tries}")
    if good / args.tries < 0.6:
        out("This model struggles with the output format. Try: a larger model, "
            "or simpler/smaller plan steps.")
        return EXIT_FAILED
    return EXIT_OK


def cmd_run(args, out, *, llm_factory, guard_factory, **_):
    cfg = load_config()
    plan = load_plan(args.plan)
    _check_test_command(cfg, plan)
    ws_path = args.workspace or plan.workspace
    if not ws_path:
        out("no workspace: pass --workspace DIR or add 'workspace' to the plan")
        return EXIT_USAGE
    ws_path = Path(ws_path).expanduser()
    if args.dry_run:
        out(f"[dry run] would build '{plan.title}' in {ws_path}")
        for i, s in enumerate(plan.steps, 1):
            out(f"  {i}. {s.title}  ->  {s.commit}")
        return EXIT_OK

    ws = Workspace(ws_path, cfg.max_file_bytes)
    log, close = _make_log(out, ws.root / ".codeagent" / "agent.log")
    stopper = _Stopper()
    stopper.install()
    try:
        runner = Runner(cfg, plan, ws, llm_factory(cfg),
                        guard=guard_factory(cfg, ws.root) if guard_factory else None,
                        log=log, should_stop=stopper, live=LivePrinter() if args.live else None)
        report = runner.run()
    except WorkspaceBusy as e:
        out(str(e))
        return EXIT_BUSY
    except KeyboardInterrupt:
        out("aborted")
        return 130
    finally:
        stopper.restore()
        close()
    out(report.summary())
    return {"done": EXIT_OK, "failed": EXIT_FAILED, "paused": EXIT_PAUSED}[report.outcome]


# ------------------------------------------------------------------ entry point

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="codeagent", description="Offline plan-driven coding agent.")
    p.add_argument("--version", action="version", version=f"codeagent {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("template", help="print (or write) a plan template")
    sp.add_argument("--out"); sp.add_argument("--force", action="store_true")
    sp = sub.add_parser("check", help="validate a plan without running it")
    sp.add_argument("plan")
    sp = sub.add_parser("doctor", help="check model server, model, git and system resources")
    sp = sub.add_parser("smoke", help="test whether the model follows the output format")
    sp.add_argument("--tries", type=int, default=5)
    sp = sub.add_parser("run", help="build a plan")
    sp.add_argument("plan"); sp.add_argument("--workspace"); sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--live", action="store_true",
                    help="show each prompt and the model's raw output (thinking too) as it streams; terminal only, not saved")
    return p


COMMANDS = {"template": cmd_template, "check": cmd_check, "doctor": cmd_doctor,
            "smoke": cmd_smoke, "run": cmd_run}


def _silence_stdout() -> None:
    """After a closed pipe, stop Python complaining again while it flushes at exit."""
    try:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    except (OSError, ValueError):
        pass


def main(argv=None, *, llm_factory=make_client, guard_factory=None, out=print) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args, out, llm_factory=llm_factory, guard_factory=guard_factory)
    except (ConfigError, PlanError) as e:
        out(f"error: {e}")
        return EXIT_USAGE
    except BrokenPipeError:                      # e.g. `codeagent template | head`
        return EXIT_OK


def entry() -> None:
    """Console-script entry point (the `codeagent` command)."""
    code = main()
    try:
        sys.stdout.flush()
    except BrokenPipeError:                      # the reader (e.g. `head`) went away
        _silence_stdout()
    sys.exit(code)


if __name__ == "__main__":
    entry()
