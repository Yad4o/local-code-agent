"""codeagent: an offline, resource-safe coding agent for small local models.

One module on purpose. Config, plan, prompts, log and the agent loop live here
until there is a real reason to split them. The harness decides what happens
next; the model only answers narrow questions.
"""
import argparse
import ast
import builtins
import contextlib
import copy
import fcntl
import getpass
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


def _is_bare_code(text: str) -> bool:
    """True when a reply with no fence is plain Python (so it can be used as the only file), not prose."""
    body = textwrap.dedent(text).strip("\n")
    if not body.strip():
        return False
    try:
        ast.parse(body)
    except SyntaxError:
        return False
    return True


def _recipe_requirements(description: str) -> list[tuple[str, str]]:
    """(source line, ast key) for every numbered/bulleted description line that is a Python statement,
    one key per statement the line asks for (an `if` asks for its test and for each statement in its one-line body)."""
    out: list[tuple[str, str]] = []
    for raw in description.splitlines():
        m = re.match(r"\s*(?:\d+[.)]|-)\s+(.*\S)\s*$", raw)
        if not m:
            continue
        text = m.group(1).strip()
        src = text + " pass" if text.endswith(":") else text
        try:
            nodes = ast.parse(src).body
        except SyntaxError:
            continue
        if len(nodes) != 1:
            continue
        for key in _statement_keys(nodes[0]):
            out.append((text, key))
    return out


def _statement_keys(node: ast.stmt) -> list[str]:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return []                                        # imports are hoisted to the top of the module, not in the body
    if isinstance(node, ast.If):
        keys = ["test:" + ast.dump(node.test)]
        for child in node.body:
            if not isinstance(child, ast.Pass):
                keys += _statement_keys(child)
        return keys
    if isinstance(node, ast.Expr):
        return ["expr:" + ast.dump(node.value)]
    return [ast.dump(node)]


def missing_recipe_lines(description: str, body: str, limit: int = 3) -> list[str]:
    """Recipe lines (code the description asks for) that the body does not contain, in recipe order.
    A tiny model often drops or rewrites one line; naming it is the most useful thing to tell it."""
    try:
        tree = ast.parse(textwrap.dedent(body))
    except SyntaxError:
        return []
    have: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            have.add("test:" + ast.dump(node.test))
        if isinstance(node, ast.Expr):
            have.add("expr:" + ast.dump(node.value))
        if isinstance(node, ast.stmt):
            have.add(ast.dump(node))
    missing: list[str] = []
    for text, key in _recipe_requirements(description):
        if key not in have and text not in missing:
            missing.append(text)
    return missing[:limit]


def _missing_lines_note(description: str, body: str) -> str | None:
    lines = missing_recipe_lines(description, body)
    if not lines:
        return None
    return ("these lines from the numbered steps are missing from your body; the steps must be followed exactly, "
            "so write them as they are:\n" + "\n".join(f"    {l}" for l in lines))


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
    think_body_after: int = 0                   # a body that failed this many attempts WITHOUT thinking is tried with thinking
                                                # (still bounded by max_think_chars); 0 = never
    rescue: bool = True                         # when the attempts run out: try the recipe as code, then a repair prompt
    rescue_attempts: int = 3                    # model calls for the repair prompt
    max_lessons: int = 8                        # rules learned from fixes that are put into every body prompt
    context_view: str = "signatures"             # helper files in a body prompt: "signatures" (def lines only) or "full" code
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
    if cfg.context_view not in ("signatures", "full"):
        raise ConfigError(f"context_view: must be 'signatures' or 'full', got {cfg.context_view!r}")
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


def _run(root: str | Path, *args: str, check: bool = True, env_extra: dict[str, str] | None = None) -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_PAGER": "cat", **(env_extra or {})}
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


def commit_paths(root: str | Path, paths: list[str] | tuple[str, ...], message: str, *,
                 literals: tuple[str, ...] = ()) -> str | None:
    """Commit exactly `paths`. Returns the short hash, or None if nothing changed. Refuses to commit anything
    that looks like a secret (a token, a private key, a .env file)."""
    bad = scan_files(root, list(paths), literals)
    if bad:
        raise GitError("refusing to commit, this looks like a secret:\n  " + "\n  ".join(bad[:5]))
    _run(root, "add", "--", *paths)
    if not _run(root, "status", "--porcelain", "--", *paths):
        return None
    cfg = ["-c", "commit.gpgsign=false"]              # a signing prompt would hang forever
    if not _has_identity(root):
        cfg += ["-c", "user.name=codeagent", "-c", "user.email=codeagent@localhost"]
    _run(root, *cfg, "commit", "-q", "--no-verify", "-m", message, "--", *paths)
    return _run(root, "rev-parse", "--short", "HEAD")


# ==========================================================================
# secrets
# `.env` loading (the token never goes into os.environ, so generated code and test runs cannot read it) and a
# scanner that refuses to commit or push anything that looks like a secret.
# ==========================================================================

def parse_dotenv(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] in "\"'" and value.endswith(value[0]):
            value = value[1:-1]
        else:
            value = re.sub(r"\s+#.*$", "", value).strip()
        out[key] = value
    return out


@dataclass
class Secrets:
    values: dict[str, str] = field(default_factory=dict)
    files: list[Path] = field(default_factory=list)          # the .env files that were read
    warnings: list[str] = field(default_factory=list)

    def get(self, *keys: str) -> str | None:
        for k in keys:
            if self.values.get(k):
                return self.values[k]
        return None

    @property
    def token(self) -> str | None:
        return self.get("GITHUB_TOKEN", "GH_TOKEN")

    def scrub(self, text: str) -> str:
        """Never let the token appear in a log line or error message."""
        tok = self.token
        return text.replace(tok, "***") if tok and len(tok) >= 8 else text


def load_secrets(explicit: str | Path | None = None, project_dir: str | Path | None = None,
                 environ: dict[str, str] | None = None, home: Path | None = None) -> Secrets:
    """Priority, high to low: the --env file, real environment variables, $CODEAGENT_ENV, a .env next to
    project.json, a .env in the current folder, ~/.codeagent.env."""
    environ = os.environ if environ is None else environ
    home = home or Path.home()
    s = Secrets()

    def read(p: Path) -> None:
        if not p.is_file():
            return
        try:
            s.values.update(parse_dotenv(p.read_text(encoding="utf-8")))
        except OSError:
            return
        if p not in s.files:
            s.files.append(p)
        try:
            if os.name == "posix" and p.stat().st_mode & 0o077:
                s.warnings.append(f"{p} can be read by other users; run: chmod 600 {p}")
        except OSError:
            pass

    for p in [home / ".codeagent.env", Path.cwd() / ".env", *([Path(project_dir) / ".env"] if project_dir else [])]:
        read(p)
    if environ.get("CODEAGENT_ENV"):
        read(Path(environ["CODEAGENT_ENV"]).expanduser())
    for key in ("GITHUB_TOKEN", "GH_TOKEN", "GITHUB_USER", "GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL",
                "GITHUB_API_URL", "GITHUB_GIT_URL_TEMPLATE"):
        if environ.get(key):
            s.values[key] = environ[key]
    if explicit:
        read(Path(explicit).expanduser())
    return s


SECRET_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = [
    ("GitHub token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b")),
    ("GitHub fine-grained token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b")),
    ("AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("private key", re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("API key", re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{24,}\b")),
]
_HARDCODED = re.compile(r"""(?i)\b(?:api[_-]?key|secret|token|passwd|password)\b['"]?\s*[:=]\s*['"]([^'"\s]{20,})['"]""")
BLOCKED_FILES = re.compile(
    r"(?:^|/)(?:\.env(?:\.(?!example$|sample$|template$)[^/]*)?|[^/]+\.pem|[^/]+\.key|[^/]+\.p12|[^/]+\.pfx|"
    r"id_(?:rsa|dsa|ecdsa|ed25519)|\.netrc|\.npmrc|\.pypirc|credentials\.json|service[-_]?account[^/]*\.json|"
    r"secrets?\.(?:json|ya?ml|toml))$")
SCAN_MAX_BYTES = 1_000_000


def _mask(s: str) -> str:
    return s[:4] + "…" if len(s) > 4 else "…"


def scan_text(text: str, literals: tuple[str, ...] = ()) -> list[tuple[int, str]]:
    """(line number, what it looks like) for every secret-looking thing in `text`."""
    found: list[tuple[int, str]] = []
    for no, line in enumerate(text.splitlines(), 1):
        for kind, rx in SECRET_PATTERNS:
            m = rx.search(line)
            if m:
                found.append((no, f"{kind} ({_mask(m.group(0))})"))
        m = _HARDCODED.search(line)
        if m and re.search(r"\d", m.group(1)) and re.search(r"[A-Za-z]", m.group(1)):
            found.append((no, f"hard-coded secret ({_mask(m.group(1))})"))
        for lit in literals:
            if lit and len(lit) >= 8 and lit in line:
                found.append((no, "your own token"))
    return found


def blocked_name(rel: str) -> bool:
    return bool(BLOCKED_FILES.search(rel.replace("\\", "/")))


def scan_files(root: str | Path, rels: list[str] | tuple[str, ...], literals: tuple[str, ...] = ()) -> list[str]:
    """Problems with committing `rels` (paths under `root`); an empty list means they are safe."""
    root = Path(root)
    problems: list[str] = []
    for rel in rels:
        if blocked_name(rel):
            problems.append(f"{rel}: a file that normally holds secrets ({'.env' if '.env' in rel else 'key/credentials'}); never committed")
            continue
        p = root / rel
        try:
            if not p.is_file() or p.stat().st_size > SCAN_MAX_BYTES:
                continue
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        problems += [f"{rel}:{no}: looks like a secret: {what}" for no, what in scan_text(text, literals)]
    return problems


def scan_history(root: str | Path, literals: tuple[str, ...] = ()) -> list[str]:
    """Scan every commit of every branch (added lines and file names), so a secret that was committed and
    deleted again is still found before anything is pushed."""
    problems: list[str] = []
    names = _run(root, "log", "--all", "--name-only", "--format=", check=False).splitlines()
    problems += [f"{n}: a file that normally holds secrets, found in the history" for n in sorted(set(names)) if n and blocked_name(n)]
    patch = _run(root, "log", "--all", "-p", "--no-color", "--format=", check=False)
    current = "?"
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
        elif line.startswith("+") and not line.startswith("+++"):
            for _, what in scan_text(line[1:], literals):
                msg = f"{current}: looks like a secret in the history: {what}"
                if msg not in problems:
                    problems.append(msg)
    return problems


GITIGNORE = """\
# secrets: never commit these
.env
.env.*
!.env.example
*.pem
*.key
*.p12
id_rsa*
.netrc
.npmrc
.pypirc
credentials.json

# build and tool output
.codeagent/
__pycache__/
*.pyc
.pytest_cache/
.venv/
venv/
.DS_Store

# your data (the web app keeps it here)
data.json
.data-*
"""


# ==========================================================================
# github
# Creates the private repo, sets the commit identity to YOUR GitHub account, and pushes with your token.
# The token is only ever handed to git through a throw-away GIT_ASKPASS helper: it is never in a URL, a
# remote, a command line, a log line or the process environment of the tests.
# ==========================================================================

class GitHubError(Exception):
    pass


class RepoExists(GitHubError):
    pass


class GitHub:
    def __init__(self, token: str, api_url: str = "https://api.github.com", *, timeout: float = 30):
        self.token, self.api_url, self.timeout = token, api_url.rstrip("/"), timeout

    def _scrub(self, text: str) -> str:
        return text.replace(self.token, "***") if self.token and len(self.token) >= 8 else text

    def _call(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict, dict[str, str]]:
        req = urllib.request.Request(
            self.api_url + path, method=method, data=None if body is None else json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "codeagent", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw, status, headers = r.read().decode("utf-8", "replace"), r.status, r.headers
        except urllib.error.HTTPError as e:
            raw, status, headers = e.read().decode("utf-8", "replace"), e.code, e.headers
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise GitHubError(f"cannot reach GitHub: {self._scrub(str(e))}") from e
        try:
            data = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            data = {"message": raw[:200]}
        return status, data if isinstance(data, dict) else {"items": data}, {k.lower(): v for k, v in (headers or {}).items()}

    def whoami(self) -> dict:
        status, data, headers = self._call("GET", "/user")
        if status == 401:
            raise GitHubError("GitHub rejected the token (401 bad credentials). Make a new token with the `repo` scope "
                              "(GitHub > Settings > Developer settings > Personal access tokens) and run: codeagent setup")
        if status != 200:
            raise GitHubError(f"GitHub answered {status}: {self._scrub(str(data.get('message', '')))}")
        login = data.get("login") or ""
        return {"login": login, "id": data.get("id"), "name": data.get("name") or login,
                "email": data.get("email") or f"{data.get('id')}+{login}@users.noreply.github.com",
                "scopes": [s.strip() for s in headers.get("x-oauth-scopes", "").split(",") if s.strip()]}

    def create_repo(self, name: str, description: str = "") -> dict:
        status, data, _ = self._call("POST", "/user/repos", {"name": name, "description": description[:340],
                                                              "private": True, "auto_init": False})
        if status == 201:
            if data.get("private") is not True:
                raise GitHubError("GitHub did not create a PRIVATE repository, so nothing will be pushed to it. "
                                  "Delete that repository on github.com and try again.")
            return data
        text = json.dumps(data)
        if status == 422 and "already exists" in text:
            raise RepoExists(name)
        if status in (401, 403):
            raise GitHubError(f"GitHub refused to create the repository ({status}). The token needs the `repo` scope "
                              f"(a classic token), then run: codeagent setup")
        raise GitHubError(f"GitHub could not create the repository ({status}): {self._scrub(str(data.get('message', '')))}")

    def create_unique_repo(self, name: str, description: str = "", tries: int = 6) -> dict:
        """Create a PRIVATE repo; if the name is taken use name-2, name-3 ... (never touches an existing repo)."""
        for n in range(1, tries + 1):
            candidate = name if n == 1 else f"{name}-{n}"
            try:
                return self.create_repo(candidate, description)
            except RepoExists:
                continue
        raise GitHubError(f"the repository names {name}, {name}-2 ... {name}-{tries} all exist already; "
                          f"set another \"name\" in project.json")


def repo_name(project_name: str) -> str:
    """A good GitHub repository name: lower case, words joined with dashes."""
    slug = re.sub(r"[^a-z0-9._]+", "-", project_name.strip().lower()).strip("-._")
    return (slug or "project")[:90]


def set_identity(root: str | Path, name: str, email: str) -> None:
    """Commits in this repository are made as YOU (local config only; your global git config is left alone)."""
    _run(root, "config", "user.name", name)
    _run(root, "config", "user.email", email)
    _run(root, "config", "commit.gpgsign", "false")


def remote_url(secrets: Secrets, owner: str, repo: str) -> str:
    template = secrets.get("GITHUB_GIT_URL_TEMPLATE") or "https://github.com/{owner}/{repo}.git"
    return template.format(owner=owner, repo=repo)


@contextlib.contextmanager
def _askpass_env(user: str, token: str):
    d = Path(tempfile.mkdtemp(prefix="codeagent-askpass-"))
    try:
        script = d / "askpass.sh"
        script.write_text('#!/bin/sh\ncase "$1" in\n  *sername*) printf "%s\\n" "$CODEAGENT_PUSH_USER" ;;\n'
                          '  *) printf "%s\\n" "$CODEAGENT_PUSH_TOKEN" ;;\nesac\n', encoding="utf-8")
        script.chmod(0o700)
        yield {"GIT_ASKPASS": str(script), "GIT_TERMINAL_PROMPT": "0", "CODEAGENT_PUSH_USER": user,
               "CODEAGENT_PUSH_TOKEN": token}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def push_ref(root: str | Path, url: str, branch: str, secrets: Secrets, user: str) -> None:
    """Push `branch` to `url` as YOU. Never forces; stored git credentials are ignored on purpose."""
    try:
        with _askpass_env(user, secrets.token or "") as env:
            _run(root, "-c", "credential.helper=", "push", url, f"refs/heads/{branch}:refs/heads/{branch}", env_extra=env)
    except GitError as e:
        raise GitError(secrets.scrub(str(e))) from None


def preflight_push(root: str | Path, literals: tuple[str, ...] = ()) -> None:
    """Refuse to push a history that contains anything that looks like a secret."""
    bad = scan_history(root, literals)
    if bad:
        raise GitError("not pushing: the history contains secrets:\n  " + "\n  ".join(bad[:8]) +
                       "\nRemove them from the history first, then run the command again.")


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
        bad = src.splitlines()[e.lineno - 1].strip() if e.lineno and 0 < e.lineno <= len(src.splitlines()) else ""
        where = f"the line `{bad}`" if bad else f"line {e.lineno} of the module"
        return None, f"the body is not valid Python inside `def {step.signature}:` ({e.msg}, {where})"
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


def signature_view(source: str) -> str | None:
    """`def name(args): ...` for every top-level function of a module (first docstring line kept), or None when
    there is none or the source does not parse. A tiny model shown a helper's whole body tends to paste it
    instead of calling it; the signature is all it needs to call it."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    blocks = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            head = ("async def " if isinstance(node, ast.AsyncFunctionDef) else "def ") + f"{node.name}({ast.unparse(node.args)})"
            if node.returns is not None:
                head += f" -> {ast.unparse(node.returns)}"
            lines = [head + ":"]
            doc = ast.get_docstring(node)
            if doc:
                lines.append(f'    """{doc.splitlines()[0]}"""')
            lines.append("    ...")
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks) or None


def _body_task_parts(plan: Plan, step: Step, tests: dict[str, str], existing: dict[str, str | None],
                     budget: int, context_view: str) -> list[str]:
    """What every body prompt starts with: the task, the def line, the examples, callable helpers."""
    name = _function_name(step)
    parts = [_header(plan, step),
             f"The function line is already written. Write ONLY its body:\n\ndef {step.signature}:\n    ..."]
    if step.examples:
        parts.append("The tests check exactly these examples:\n" + "\n".join(_example_line(name, e)[:300] for e in step.examples))
    else:
        parts.append("These tests must pass:\n" + _fit(dict(tests), budget // 2))
    ctx = {p: existing.get(p) for p in step_context(step)}
    if any(v is not None for v in ctx.values()):
        header = "Related files (read-only):"
        if context_view == "signatures":
            shown = {p: (signature_view(v) if v else None) for p, v in ctx.items()}
            if any(shown.values()):
                ctx = {p: shown[p] or v for p, v in ctx.items()}
                header = ("Functions you can call (read-only; only their signatures are shown, so call them "
                          "and do not rewrite their code):")
        parts.append(header + "\n" + _fit(ctx, budget // 2))
    ready = [ln.rsplit(" import ", 1)[1] for ln in context_imports(step, existing)]
    if ready:
        parts.append("Already imported for you: " + ", ".join(ready) + ". Do not write imports for them.")
    return parts


def _lessons_part(lessons: tuple[str, ...] | list[str]) -> str | None:
    return ("Rules learned from earlier mistakes (follow them):\n" + "\n".join(f"- {l}" for l in lessons)) if lessons else None


def build_body_prompt(plan: Plan, step: Step, tests: dict[str, str], existing: dict[str, str | None],
                      budget: int, test_output: str | None, feedback: str | None,
                      context_view: str = "signatures", lessons: tuple[str, ...] | list[str] = ()) -> list[dict]:
    """Skeleton mode: the model sees the task, the def line, the examples, how its last body failed, and
    what the tests said. Never the pytest file, the package __init__, or a whole module with a def line."""
    path, name = module_path(step), _function_name(step)
    parts = _body_task_parts(plan, step, tests, existing, budget, context_view)
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
    if _lessons_part(lessons):
        parts.append(_lessons_part(lessons))
    parts.append("Reply with ONLY this, the body lines and nothing else (no def line, no docstring, "
                 f"no comments, no explanation):\nFILE: {path}\n```python\n<body lines>\n```")
    return [{"role": "system", "content": SYSTEM_BODY}, {"role": "user", "content": "\n\n".join(parts)}]


SYSTEM_RESCUE = ("You repair one small Python function. You get its numbered steps, a draft body and the report of "
                 "the tests it fails. Rewrite the body so that every step is followed exactly as written and every example "
                 "passes. Change as little as you can. Reply with ONLY the corrected body, in exactly this format and "
                 "nothing else:\n\nFILE: path/to/file.py\n```python\n<body lines only>\n```")


def build_rescue_prompt(plan: Plan, step: Step, tests: dict[str, str], existing: dict[str, str | None], budget: int,
                        draft: str, test_output: str | None, context_view: str = "signatures",
                        lessons: tuple[str, ...] | list[str] = ()) -> list[dict]:
    """The different prompt used when the normal attempts failed: not 'write it', but 'fix this draft'."""
    path, name = module_path(step), _function_name(step)
    parts = _body_task_parts(plan, step, tests, existing, budget, context_view)
    parts.append("This draft body does not pass the tests yet:\n```python\n" + (draft or "pass")[:budget // 2] + "\n```")
    if test_output:
        parts.append("What the tests reported for the draft:\n" + _name_the_calls(condense_test_output(test_output), name, step.examples))
    note = _missing_lines_note(step.description, draft or "")
    if note:
        parts.append("Also: " + note)
    if _lessons_part(lessons):
        parts.append(_lessons_part(lessons))
    parts.append("Fix the draft. Keep every line of the numbered steps. Reply with ONLY the corrected body "
                 f"(no def line, no comments, no explanation):\nFILE: {path}\n```python\n<body lines>\n```")
    return [{"role": "system", "content": SYSTEM_RESCUE}, {"role": "user", "content": "\n\n".join(parts)}]


_RECIPE_ROW = re.compile(r"^(\s*)(?:\d+[.)]|-)\s+(.*\S)\s*$")


def compile_recipe(description: str) -> str | None:
    """The numbered and bulleted lines of a description as a Python body, when EVERY one of them is already
    code (`if x < 0: raise ValueError`, `For each item in cart:` ...). None as soon as one line is English."""
    rows = [(len(m.group(1)), m.group(2).strip()) for m in map(_RECIPE_ROW.match, description.splitlines()) if m]
    if not rows:
        return None
    stack: list[int] = []
    lines: list[str] = []
    for indent, text in rows:
        while stack and stack[-1] >= indent:
            stack.pop()
        m = re.fullmatch(r"For each (\w+) in (.+):", text)
        text = f"for {m.group(1)} in {m.group(2)}:" if m else text
        lines.append("    " * len(stack) + text)
        stack.append(indent)
    body = "\n".join(lines)
    try:
        ast.parse("def _f():\n" + textwrap.indent(body, "    "))
    except SyntaxError:
        return None
    return body


def _slug_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:80] or "step"


def global_lessons_path() -> Path:
    return Path(os.environ.get("CODEAGENT_LESSONS") or Path.home() / ".config" / "codeagent" / "lessons.json")


def load_lessons(root: str | Path, limit: int = 8) -> list[str]:
    """Rules learned from `codeagent fix`: this project's and the global ones (project rules last = freshest)."""
    out: list[str] = []
    for path in (global_lessons_path(), Path(root) / DIR / "lessons.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out += [l for l in data.get("lessons", []) if isinstance(l, str) and l.strip() and l not in out]
    return out[-limit:] if limit else []


def override_path(root: str | Path, step: Step) -> Path:
    return Path(root) / DIR / "overrides" / f"{_slug_title(step.title)}.py"


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
                  budget: int, test_output: str | None, feedback: str | None,
                  context_view: str = "signatures", lessons: tuple[str, ...] | list[str] = ()) -> list[dict]:
    if step.signature:
        return build_body_prompt(plan, step, tests, existing, budget, test_output, feedback, context_view, lessons)
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
    via: str = ""                     # how a step that needed help got done: recipe | rescue prompt | your fix
    detail: dict = field(default_factory=dict)       # for a failed step: the bodies tried, test output, notes


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
            extra = (f"commit {r.commit}, {r.attempts} model calls" + (f", rescued: {r.via}" if r.via else "")
                     if r.status == "done" else r.reason)
            lines.append(f"  [{r.status}] {r.title}" + (f"  ({extra})" if extra else ""))
        return "\n".join(lines)


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items)


class Runner:
    def __init__(self, cfg: Config, plan: Plan, ws: Workspace, llm: LLMClient, *,
                 guard=None, run_cmd=run_command, log: Callable[[str], None] | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 should_stop: Callable[[], bool] = lambda: False, live: LivePrinter | None = None,
                 branch: str | None = None, plan_path: str | Path | None = None):
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
        self.branch, self.plan_path = branch, (str(plan_path) if plan_path else None)
        self.lessons = load_lessons(ws.root, cfg.max_lessons)
        self._tried: list[str] = []          # bodies the model produced for the current step
        self._notes: list[str] = []          # what went wrong, attempt by attempt
        self._last_output = ""
        self._via = ""

    def _note(self, msg: str) -> None:
        self._notes.append(msg)
        self.log("   " + msg)

    # ------------------------------------------------------------------ run
    def run(self) -> RunReport:
        with WorkspaceLock(self.ws.root):
            ensure_repo(self.ws.root)
            branch = self.branch or f"agent/{plan_slug(self.plan)}"
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
        self._tried, self._notes, self._last_output, self._via = [], [], "", ""
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
            return StepResult(step.title, "done", self.attempts, commit=sha, via=self._via)
        except _Pause as p:
            self.ws.restore(snapshot)
            return StepResult(step.title, "paused", self.attempts, reason=str(p))
        except (_Fail, GitError, WorkspaceError) as e:
            self.ws.restore(snapshot)
            return StepResult(step.title, "failed", self.attempts, reason=str(e),
                              detail={"bodies": self._tried[-3:], "output": self._last_output, "notes": self._notes[-8:]})

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
        cannot_think = False                             # a call that thought and never answered: never think again
        if step.signature:
            gone = missing_named_files(step, self._read_many(step_context(step)))
            if gone:
                raise _Fail(f"the step description names {', '.join(gone)}, but it is missing or empty in the "
                            f"workspace; an earlier step must create it first (check that it passed and was committed)")
            output, satisfied = self._reset_skeleton(step, tests, output)
            if satisfied:
                return
            override = override_path(self.ws.root, step)
            if override.is_file():                      # a body supplied through `codeagent fix`
                ok, out, _ = self._try_reply(step, tests, f"FILE: {module_path(step)}\n```python\n{override.read_text(encoding='utf-8')}\n```")
                if ok:
                    self._via = "your fix"
                    return
                self._note("the body from your fix did not pass the tests; trying the normal way")
                output, satisfied = self._reset_skeleton(step, tests, output)
        for n in range(1, self.cfg.max_attempts + 1):
            existing = self._read_many((*step.files, *step_context(step)))
            if fresh:                                    # hide only the model's own old code, never the related files
                existing.update({p: None for p in step.files})

            text = self._ask(build_impl_prompt(self.plan, step, tests, existing,
                                               self.cfg.max_context_chars, output, feedback, self.cfg.context_view,
                                               self.lessons),
                             step=step, phase="impl", attempt=n,
                             num_predict=self.cfg.num_predict_body if step.signature else self.cfg.num_predict,
                             stop=BODY_STOP if step.signature else None, bump=0.15 * repeats,
                             think=False if (quiet and (cannot_think or not self.cfg.think_body_after or n < self.cfg.think_body_after)) else None)
            fresh = False
            if getattr(text, "no_answer", ""):
                quiet = cannot_think = True
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
                self._note(f"impl attempt {n}: {problems[0]}")
                continue
            key = "\n".join(f"{p}\n{(function_body(files[p], _function_name(step)) if step.signature and p == module_path(step) else None) or files[p]}"
                            for p in sorted(files))
            if key in seen:                              # a tiny model often copies its last answer verbatim
                repeats += 1
                fresh = True
                feedback = (f"this is identical to your attempt {seen[key]}, which already failed the tests. "
                            f"Write a DIFFERENT solution; do not copy your earlier code.")
                if step.signature:
                    note = _missing_lines_note(step.description, function_body(files[module_path(step)], _function_name(step)) or "")
                    feedback += "\nAlso: " + note if note else ""
                self._note(f"impl attempt {n}: same code as attempt {seen[key]}; not re-running, asking for a different one")
                continue
            seen[key] = n
            if step.signature:
                self._tried.append(function_body(files[module_path(step)], _function_name(step)) or "")
            self._write(files)
            self._write(tests)
            problems = self._static_check(list(files))
            if problems:
                feedback = _bullets(problems)
                self._note(f"impl attempt {n}: static check: {problems[0]}")
                continue
            feedback = None
            res = self._run_tests()
            if res.ok:
                return
            broken = diagnose_broken_tests(res.output)
            if broken:
                raise _TestsBroken(broken)
            output = self._last_output = res.output
            if step.signature:                           # name the recipe lines the body left out or rewrote
                feedback = _missing_lines_note(step.description, function_body(files[module_path(step)], _function_name(step)) or "")
            self._note(f"impl attempt {n}: tests still failing")
        if step.signature and self.cfg.rescue:
            self._last_output = output
            via = self._rescue(step, tests, output)
            if via:
                self._via = via
                return
            output = self._last_output
        tail = output.strip().splitlines()[-1] if output.strip() else ""
        raise _Fail(f"tests still failing after {self.cfg.max_attempts} attempts"
                    + (f" and {self.cfg.rescue_attempts} repair tries" if step.signature and self.cfg.rescue else "")
                    + f". Last line: {tail}")

    # ------------------------------------------------------------------ the rescue ladder
    def _try_reply(self, step: Step, tests: dict[str, str], text: str) -> tuple[bool, str, str | None]:
        """Apply a reply (a model's, or a body the harness built) the way the main loop does.
        Returns (tests passed, test output or the problem, the body)."""
        files, problems = self._collect(text, (module_path(step),), check_syntax=False)
        if not problems:
            inits = {f: self._read_many([f])[f] for f in step.files if f.endswith("__init__.py")}
            files, problems = assemble_files(step, files, inits, self._read_many(step_context(step)))
        if not problems:
            problems += self._lint_impl(step, files)
        if not problems and self.cfg.tidy_code:
            files = {p: tidy_python(c) if p.endswith(".py") else c for p, c in files.items()}
        if problems:
            return False, _bullets(problems), None
        body = function_body(files[module_path(step)], _function_name(step))
        self._write(files)
        self._write(tests)
        problems = self._static_check(list(files))
        if problems:
            return False, _bullets(problems), body
        res = self._run_tests()
        if res.ok:
            return True, res.output, body
        broken = diagnose_broken_tests(res.output)
        if broken:
            raise _TestsBroken(broken)
        return False, res.output, body

    def _rescue(self, step: Step, tests: dict[str, str], output: str) -> str | None:
        """The normal attempts failed. 1) the recipe itself, if every line of it is code; 2) a different prompt:
        'repair this draft'. Returns how it was solved, or None."""
        path = module_path(step)
        draft = self._tried[-1] if self._tried else ""
        body = compile_recipe(step.description)
        if body:
            self._note("trying the steps of the description as code")
            ok, out, got = self._try_reply(step, tests, f"FILE: {path}\n```python\n{body}\n```")
            if ok:
                return "recipe"
            draft, output = got or body, out
            self._note("the steps as code do not pass the tests either")
        for n in range(1, self.cfg.rescue_attempts + 1):
            existing = self._read_many((*step.files, *step_context(step)))
            text = self._ask(build_rescue_prompt(self.plan, step, tests, existing, self.cfg.max_context_chars, draft, output,
                                                 self.cfg.context_view, self.lessons),
                             step=step, phase="repair", attempt=n, num_predict=self.cfg.num_predict_body, stop=BODY_STOP,
                             bump=0.1 * (n - 1), think=False if (self.cfg.think and not self.cfg.think_body) else None)
            ok, out, got = self._try_reply(step, tests, text)
            if ok:
                return "repair prompt"
            self._note(f"repair attempt {n}: " + (out.strip().splitlines()[-1] if out.strip() else "no answer"))
            if got:
                self._tried.append(got)
                draft, output = got, out
            self._last_output = out
        return None

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
        if not parsed.blocks and not problems and len(targets) == 1 and "```" not in str(text) and _is_bare_code(str(text)):
            parsed.blocks = [FileBlock(targets[0], str(text).strip("\n"))]         # bare code, no fence at all: still unambiguous
            self.log(f"   fixed up: the reply was bare code without a fence; used it as {targets[0]}")
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


def lint_description(text: str) -> list[str]:
    """Warnings about description lines a small model copies literally into invalid or skipped code.
    Only advice: a plan with warnings still runs."""
    out: list[str] = []
    lines = [l for l in text.splitlines() if l.strip()]
    numbered = [l for l in lines if re.match(r"\s*\d+\.\s", l)]
    loops = [l for l in lines if re.match(r"\s*(?:\d+\.|-)?\s*for each\b", l, re.I)]
    if len(numbered) > 6:
        out.append(f"{len(numbered)} numbered lines: a small model skips steps in long recipes; split this into two steps (about 5-6 lines at most)")
    if len(loops) > 1:
        out.append(f"{len(loops)} loops: keep one loop per step and split the rest into another function")
    for l in lines:
        t = l.strip()
        if re.search(r"(?<![=!<>])=(?!=)[^\n]*?\band\b\s+[\w.\[\]\"']+\s*=(?!=)", t):
            out.append(f"two assignments joined with 'and' (the model copies it as invalid Python): {t[:70]}")
        elif re.search(r"(?i)\bfor each\b[^\n]*:\s*\S", t):
            out.append(f"a loop and its body on one line (write the loop on its own line, its statements as bullets below): {t[:70]}")
        elif re.match(r"(?:\d+\.|-)\s", t) and re.search(r";\s*\S", t):
            out.append(f"statements joined with ';' (put each statement on its own line): {t[:70]}")
        elif re.match(r"(?:\d+\.|-)\s*If [^:\n]*, (?:return|raise)\b", t):
            out.append(f"an English condition is often skipped; write it as code, e.g. `if x < 0: raise ValueError`: {t[:70]}")
    return out


def lint_step(step: Step) -> list[str]:
    """Warnings about a whole step: its description (see lint_description) and its `context` entries."""
    out = lint_description(step.description)
    for entry in step.context:
        if not re.search(r"\.[A-Za-z0-9]+$", entry):
            out.append(f"context entry {entry!r} is not a file path; write it like pkg/module.py")
    return out


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
        for w in lint_step(s):
            out(f"       warning: {w}")
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


# ==========================================================================
# consolidate
# A finished plan leaves one tiny module (and one test file) per function. This merges them into a few
# themed modules, deterministically (no model): imports are rewritten, __init__ re-exports are kept, test
# names are kept unique, and the whole test suite must still pass with the same number of tests, or
# everything is put back.
# ==========================================================================

class ConsolidateError(Exception):
    pass


MARKER = "consolidated.json"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "core"


def suggest_groups(plans: list[Plan]) -> dict:
    """A starter groups file: one new module per plan, named from the plan title."""
    package = None
    modules: list[list[str]] = []
    for plan in plans:
        mods = []
        for step in plan.steps:
            head, _, _ = step.function.partition(":")
            parts = head.split(".")
            if len(parts) != 2:
                raise ConsolidateError(f"step {step.title!r}: function {step.function!r} is not package.module:name")
            package = package or parts[0]
            if parts[0] != package:
                raise ConsolidateError("the plans use more than one package; consolidate one package at a time")
            if parts[1] not in mods:
                mods.append(parts[1])
        modules.append(mods)
    words = [re.sub(r"[^A-Za-z0-9 ]+", " ", p.title).split() for p in plans]
    common = 0
    while len(plans) > 1 and all(len(w) > common + 1 for w in words) and len({w[common].lower() for w in words}) == 1:
        common += 1
    names: list[str] = []
    for w in words:
        name = _slug(" ".join(w[common:]))
        if name in names or name == package:
            name = _slug(" ".join(w))
        names.append(name)
    return {"package": package, "modules": dict(zip(names, modules))}


def load_groups(path: str | Path) -> tuple[str, dict[str, list[str]]]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ConsolidateError(f"cannot read {path}: {e}") from e
    package, modules = raw.get("package") if isinstance(raw, dict) else None, raw.get("modules") if isinstance(raw, dict) else None
    if not isinstance(package, str) or not package.isidentifier():
        raise ConsolidateError("groups file: 'package' must be the package name, e.g. \"expense_manager\"")
    if not isinstance(modules, dict) or not modules:
        raise ConsolidateError("groups file: 'modules' must map each new module name to a list of old module names")
    out: dict[str, list[str]] = {}
    for new, olds in modules.items():
        if not new.isidentifier() or new == "__init__":
            raise ConsolidateError(f"groups file: {new!r} is not a usable module name")
        if not isinstance(olds, list) or not olds or not all(isinstance(o, str) and o.isidentifier() for o in olds):
            raise ConsolidateError(f"groups file: {new!r} must list one or more old module names")
        out[new] = list(dict.fromkeys(olds))
    seen: dict[str, str] = {}
    for new, olds in out.items():
        for o in olds:
            if o in seen:
                raise ConsolidateError(f"module {o!r} is in both {seen[o]!r} and {new!r}")
            seen[o] = new
    return package, out


@dataclass
class Consolidation:
    writes: dict[str, str]
    deletes: list[str]
    summary: list[tuple[str, list[str], list[str]]]      # (new module, old modules, names it now holds)
    untouched: list[str]


def _segment(lines: list[str], node: ast.stmt) -> str:
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return "\n".join(lines[start - 1:node.end_lineno])


def _defined_names(node: ast.stmt) -> list[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def _drop_unused_call_assigns(func: ast.FunctionDef | ast.AsyncFunctionDef):
    """`x = f(...)` where x is never read in the function becomes `f(...)`. Returns (new function, changed)."""
    loaded = {n.id for n in ast.walk(func) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    changed = False
    body = []
    for stmt in func.body:
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id not in loaded and isinstance(stmt.value, ast.Call)):
            body.append(ast.copy_location(ast.Expr(value=stmt.value), stmt))
            changed = True
        else:
            body.append(stmt)
    if changed:
        func = ast.fix_missing_locations(ast.copy_location(type(func)(**{**{f: getattr(func, f) for f in func._fields}, "body": body}), func))
    return func, changed


class _Imports:
    """Imports of one new module: external ones merged and de-duplicated, internal ones pointed at the new module."""

    def __init__(self) -> None:
        self.plain: list[tuple[str, str | None]] = []
        self.froms: dict[str, list[tuple[str, str | None]]] = {}
        self.raw: list[str] = []
        self.internal: dict[str, list[str]] = {}

    def add_plain(self, name: str, asname: str | None) -> None:
        if (name, asname) not in self.plain:
            self.plain.append((name, asname))

    def add_from(self, module: str, name: str, asname: str | None) -> None:
        bucket = self.froms.setdefault(module, [])
        if (name, asname) not in bucket:
            bucket.append((name, asname))

    def add_internal(self, group: str, name: str) -> None:
        bucket = self.internal.setdefault(group, [])
        if name not in bucket:
            bucket.append(name)

    @staticmethod
    def _from_line(module: str, names: list[str]) -> str:
        one = f"from {module} import {', '.join(names)}"
        if len(one) <= 99:
            return one
        return f"from {module} import (\n" + "".join(f"    {n},\n" for n in names) + ")"

    def render(self, package: str, own_group: str) -> str:
        external = [f"import {n}" + (f" as {a}" if a else "") for n, a in sorted(self.plain, key=lambda t: (t[0], t[1] or ""))]
        external += [self._from_line(m, [n + (f" as {a}" if a else "") for n, a in sorted(names, key=lambda t: t[0])])
                     for m, names in sorted(self.froms.items())]
        external += [r for r in self.raw if r not in external]
        internal = [self._from_line(f"{package}.{g}", names) for g, names in sorted(self.internal.items()) if g != own_group]
        return "\n".join(external) + ("\n\n" if external and internal else "") + "\n".join(internal)


def _internal_target(node: ast.ImportFrom, package: str) -> str | None:
    """The old module an import points at (absolute `package.x` or relative `.x`), else None."""
    if node.level == 0 and node.module and node.module.startswith(package + "."):
        rest = node.module[len(package) + 1:]
        return rest if "." not in rest else None
    if node.level == 1 and node.module and "." not in node.module:
        return node.module
    return None


def _rewrite_module_aliases(node: ast.stmt, aliases: dict[str, str]) -> tuple[ast.stmt, list[tuple[str, str]]]:
    """`alias.name` -> `name`, where alias stands for one of the old modules. Returns (new node, [(old module, name)])."""
    used: list[tuple[str, str]] = []

    class Rewrite(ast.NodeTransformer):
        def visit_Attribute(self, n: ast.Attribute):
            self.generic_visit(n)
            if isinstance(n.value, ast.Name) and n.value.id in aliases:
                used.append((aliases[n.value.id], n.attr))
                return ast.copy_location(ast.Name(id=n.attr, ctx=n.ctx), n)
            return n

    new = ast.fix_missing_locations(Rewrite().visit(copy.deepcopy(node)))
    left = [n.id for n in ast.walk(new) if isinstance(n, ast.Name) and n.id in aliases]
    if left:
        raise ConsolidateError(f"the module name {left[0]!r} is used on its own, not as {left[0]}.something, so it cannot be rewritten")
    return new, used


def _split_module(src: str, package: str, owner: dict[str, str], imports: _Imports, own_group: str, *,
                  tidy: bool, label: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Parse one source file: feed its imports into `imports`, return its other top-level statements as
    [(source text, first defined name)] and the names it defines."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise ConsolidateError(f"{label}: cannot parse ({e.msg}, line {e.lineno})") from e
    lines = src.splitlines()
    pieces: list[tuple[str, str]] = []
    names: list[str] = []
    aliases: dict[str, str] = {}                  # local name -> old module it stands for (`from pkg import mod as m`)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and ((node.level == 0 and node.module == package) or (node.level == 1 and not node.module)):
            aliases.update({a.asname or a.name: a.name for a in node.names if a.name in owner})
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith(package + ".") and a.name[len(package) + 1:] in owner:
                    if not a.asname:
                        raise ConsolidateError(f"{label}: 'import {a.name}' (without 'as') cannot be rewritten")
                    aliases[a.asname] = a.name[len(package) + 1:]
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.Import):
            for a in node.names:
                if not (a.name.startswith(package + ".") and a.name[len(package) + 1:] in owner):
                    imports.add_plain(a.name, a.asname)
        elif isinstance(node, ast.ImportFrom):
            old = _internal_target(node, package)
            if old is not None and old in owner:
                for a in node.names:
                    if a.name == "*":
                        raise ConsolidateError(f"{label}: 'import *' from {old} cannot be rewritten")
                    if owner[old] != own_group:
                        imports.add_internal(owner[old], a.name + (f" as {a.asname}" if a.asname else ""))
            elif node.level == 0 and node.module:
                for a in node.names:
                    if not (node.module == package and a.name in aliases.values()):
                        imports.add_from(node.module, a.name, a.asname)
            elif not (node.level == 1 and not node.module and all(a.name in aliases.values() for a in node.names)):
                imports.raw.append(_segment(lines, node))
        elif i == 0 and isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) \
                and isinstance(node.value.value, str):
            continue                                             # a module docstring belongs to the old file
        else:
            text = _segment(lines, node)
            if aliases:
                node, used = _rewrite_module_aliases(node, aliases)
                for old, attr in used:
                    if owner[old] != own_group:
                        imports.add_internal(owner[old], attr)
                if used:
                    text = ast.unparse(node)
            if tidy and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                new, changed = _drop_unused_call_assigns(node)
                if changed:
                    text = ast.unparse(new)
            defined = _defined_names(node)
            pieces.append((text, defined[0] if defined else ""))
            names += defined
    return pieces, names


def _find_cycle(graph: dict[str, set[str]]) -> list[str] | None:
    state: dict[str, int] = {}

    def visit(node: str, path: list[str]) -> list[str] | None:
        state[node] = 1
        for nxt in sorted(graph.get(node, ())):
            if state.get(nxt) == 1:
                return path[path.index(nxt):] + [nxt] if nxt in path else [node, nxt]
            if nxt not in state:
                found = visit(nxt, path + [nxt])
                if found:
                    return found
        state[node] = 2
        return None

    for n in sorted(graph):
        if n not in state:
            found = visit(n, [n])
            if found:
                return found
    return None


def consolidate_sources(package: str, groups: dict[str, list[str]], modules: dict[str, str], tests: dict[str, str],
                        init_text: str | None, *, tidy: bool = True) -> Consolidation:
    """Pure core. `modules` maps old module name -> source, `tests` maps old module name -> its test source."""
    owner = {old: new for new, olds in groups.items() for old in olds}
    for new, olds in groups.items():
        for old in olds:
            if old not in modules:
                raise ConsolidateError(f"{package}/{old}.py does not exist (listed under {new!r})")
        if new in modules and new not in olds:
            raise ConsolidateError(f"the new module name {new!r} is already a module that is not in that group; "
                                   f"pick another name or add {new!r} to the group")
    writes: dict[str, str] = {}
    deletes: list[str] = []
    summary: list[tuple[str, list[str], list[str]]] = []
    graph: dict[str, set[str]] = {g: set() for g in groups}
    for new, olds in groups.items():
        imports = _Imports()
        pieces: list[str] = []
        held: list[str] = []
        for old in olds:
            body, names = _split_module(modules[old], package, owner, imports, new, tidy=tidy, label=f"{package}/{old}.py")
            for n in names:
                if n in held:
                    raise ConsolidateError(f"{new!r}: the name {n!r} is defined in more than one of its modules")
            held += names
            pieces += [t for t, _ in body]
        graph[new] = set(imports.internal) - {new}
        header = imports.render(package, new)
        writes[f"{package}/{new}.py"] = (header + "\n\n\n" if header else "") + "\n\n\n".join(pieces) + "\n"
        summary.append((new, list(olds), held))
        deletes += [f"{package}/{old}.py" for old in olds if old != new]
    cycle = _find_cycle(graph)
    if cycle:
        raise ConsolidateError("these new modules would import each other in a circle: " + " -> ".join(cycle) +
                               ". Move modules between the groups so that imports only go one way.")
    # tests: one file per group; names that repeat across the merged files get the old module name in them
    for new, olds in groups.items():
        have = [o for o in olds if o in tests]
        if not have:
            continue
        imports = _Imports()
        parsed: list[tuple[str, list[tuple[str, str]]]] = []
        for old in have:
            body, _ = _split_module(tests[old], package, owner, imports, "", tidy=False, label=f"tests/test_{old}.py")
            parsed.append((old, body))
        count: dict[str, int] = {}
        for _, body in parsed:
            for _, name in body:
                count[name] = count.get(name, 0) + 1
        out_pieces: list[str] = []
        seen_names: set[str] = set()
        for old, body in parsed:
            for text, name in body:
                if name and count[name] > 1 and name.startswith("test"):
                    new_name = "test_" + old + "_" + name[len("test_"):] if name.startswith("test_") else f"test_{old}_{name}"
                    text = re.sub(rf"\bdef {re.escape(name)}\(", f"def {new_name}(", text, count=1)
                    name = new_name
                if name and name in seen_names:
                    raise ConsolidateError(f"tests/test_{new}.py: the name {name!r} would exist twice")
                seen_names.add(name)
                out_pieces.append(text)
        header = imports.render(package, "")
        writes[f"tests/test_{new}.py"] = (header + "\n\n\n" if header else "") + "\n\n\n".join(out_pieces) + "\n"
        deletes += [f"tests/test_{old}.py" for old in have if old != new]
    # __init__: the same names, now exported from the new modules
    init_lines: list[str] = []
    exports: dict[str, list[str]] = {}
    for line in (init_text or "").splitlines():
        m = re.match(rf"^from (?:\.|{re.escape(package)}\.)(\w+) import (.+)$", line.strip())
        if m and m.group(1) in owner:
            exports.setdefault(owner[m.group(1)], [])
            exports[owner[m.group(1)]] += [n.strip() for n in m.group(2).strip("() ").split(",") if n.strip()]
        elif line.strip():
            init_lines.append(line)
    for new in groups:
        names = list(dict.fromkeys(exports.get(new, [])))
        if names:
            init_lines.append(_Imports._from_line(f".{new}", names))
    writes[f"{package}/__init__.py"] = "\n".join(init_lines) + "\n"
    untouched = sorted(m for m in modules if m not in owner and m != "__init__")
    return Consolidation(writes, [d for d in dict.fromkeys(deletes) if d not in writes], summary, untouched)


def _pytest_passed(root: Path) -> tuple[int, int, str]:
    """(passed, failed+errors, last output line) of `python -m pytest -q` in `root`."""
    try:
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=str(root),
                           capture_output=True, text=True, timeout=900, stdin=subprocess.DEVNULL)
    except (subprocess.TimeoutExpired, OSError) as e:
        raise ConsolidateError(f"could not run the tests: {e}") from e
    last = (p.stdout.strip().splitlines() or [""])[-1]
    passed = int((re.search(r"(\d+) passed", last) or [0, 0])[1])
    bad = sum(int(x) for x in re.findall(r"(\d+) (?:failed|error)", last))
    return passed, bad if p.returncode else 0, last


def cmd_consolidate(args, out, **_):
    if args.suggest:
        plans = [load_plan(p) for p in args.suggest]
        out(json.dumps(suggest_groups(plans), indent=2))
        return EXIT_OK
    if not args.groups or not args.workspace:
        out("usage: codeagent consolidate GROUPS.json --workspace DIR   (or: --suggest PLAN.json [PLAN.json ...])")
        return EXIT_USAGE
    package, groups = load_groups(args.groups)
    ws = Workspace(Path(args.workspace).expanduser(), Config().max_file_bytes)
    consolidate_workspace(ws, package, groups, tidy=not args.no_tidy, commit=not args.no_commit,
                          dry_run=args.dry_run, out=out)
    return EXIT_OK


def consolidate_workspace(ws: Workspace, package: str, groups: dict[str, list[str]], *, tidy: bool = True,
                          commit: bool = True, dry_run: bool = False, out: Callable[[str], None] = print) -> str | None:
    """Merge the modules of `package` in `ws` as `groups` says; tests must pass before and after, with the same
    number of tests, or every file is put back. Returns the short hash of the commit (or None)."""
    pkg_dir = ws.root / package
    if not pkg_dir.is_dir():
        raise ConsolidateError(f"{pkg_dir} is not a directory")
    modules = {f.stem: f.read_text(encoding="utf-8") for f in sorted(pkg_dir.glob("*.py")) if f.stem != "__init__"}
    tests = {}
    for old in (o for olds in groups.values() for o in olds):
        t = ws.root / "tests" / f"test_{old}.py"
        if t.is_file():
            tests[old] = t.read_text(encoding="utf-8")
    init_path = pkg_dir / "__init__.py"
    result = consolidate_sources(package, groups, modules, tests,
                                 init_path.read_text(encoding="utf-8") if init_path.is_file() else None, tidy=tidy)
    out(f"consolidate: {package}")
    for new, olds, names in result.summary:
        out(f"  {new}.py  <-  {', '.join(olds)}   ({len(names)} name{'s' if len(names) != 1 else ''})")
    if result.untouched:
        out(f"  left as they are: {', '.join(result.untouched)}")
    n_before = sum(len(o) for o in groups.values())
    out(f"  modules {n_before} -> {len(groups)}")
    if dry_run:
        out("[dry run] nothing was written")
        return None
    git = (ws.root / ".git").exists() and commit
    if git and _run(ws.root, "status", "--porcelain"):
        raise ConsolidateError("the workspace has uncommitted changes; commit or stash them first (or use --no-commit)")
    before_passed, before_bad, before_line = _pytest_passed(ws.root)
    if before_bad or before_passed == 0:
        raise ConsolidateError(f"the tests do not pass before consolidating ({before_line}); fix that first")
    touched = [*result.writes, *result.deletes]
    snapshot = ws.snapshot(touched)
    try:
        for rel, content in result.writes.items():
            ws.write(rel, content)
        for rel in result.deletes:
            ws.delete(rel)
        after_passed, after_bad, after_line = _pytest_passed(ws.root)
    except BaseException:
        ws.restore(snapshot)
        raise
    if after_bad or after_passed != before_passed:
        ws.restore(snapshot)
        raise ConsolidateError(f"the tests changed after merging ({before_passed} passed before; now: {after_line}). "
                               "Nothing was changed.")
    out(f"  tests: {before_passed} passed before, {after_passed} passed after")
    (ws.root / DIR).mkdir(exist_ok=True)
    (ws.root / DIR / MARKER).write_text(json.dumps({"package": package, "modules": groups}, indent=2), encoding="utf-8")
    short = None
    if git:
        short = commit_paths(ws.root, touched, f"refactor: consolidate {n_before} modules into {len(groups)}")
        out(f"  committed {short} on branch {current_branch(ws.root)}   (undo: git reset --hard HEAD~1)")
    out("done. Do not rerun the plans in this workspace; to rebuild, use a new workspace.")
    return short


UI_TEMPLATE = r'''#!/usr/bin/env python3
"""A tiny local UI for __TITLE_TEXT__ (generated by codeagent). Run:  python ui.py   then open http://127.0.0.1:8000

Every public function of the package gets a form. Each value is JSON (text that is not JSON is sent as a string).
A result can be saved as a variable; type $name in any other box to use it. Only this computer can connect."""
import copy
import html
import importlib
import inspect
import json
import os
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PACKAGE = __PACKAGE__
TITLE = __TITLE_REPR__
EXAMPLES = json.loads(__EXAMPLES__)
VARS = {}
LOCK = threading.Lock()
VAR_REF = re.compile(r"^\$([A-Za-z_][A-Za-z0-9_]*)$")
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
pkg = importlib.import_module(PACKAGE)


def public_functions():
    out = {}
    for name in sorted(dir(pkg)):
        obj = getattr(pkg, name)
        if name.startswith("_") or not inspect.isfunction(obj):
            continue
        if not getattr(obj, "__module__", "").startswith(pkg.__name__):
            continue
        out[name] = obj
    return out


def describe():
    items = []
    for name, fn in public_functions().items():
        params = [{"name": p.name, "default": None if p.default is inspect.Parameter.empty else repr(p.default)}
                  for p in inspect.signature(fn).parameters.values()]
        items.append({"name": name, "params": params, "doc": (inspect.getdoc(fn) or "").split("\n")[0],
                      "examples": EXAMPLES.get(name, [])})
    return items


def _default(o):
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=repr)
    return repr(o)


def resolve(value):
    if isinstance(value, str):
        m = VAR_REF.match(value)
        if m:
            if m.group(1) not in VARS:
                raise KeyError("no saved variable $" + m.group(1))
            return copy.deepcopy(VARS[m.group(1)])
        return value
    if isinstance(value, list):
        return [resolve(v) for v in value]
    if isinstance(value, dict):
        return {k: resolve(v) for k, v in value.items()}
    return value


def call(name, args, save_as=None):
    fns = public_functions()
    if name not in fns:
        raise LookupError(name)
    with LOCK:
        try:
            values = {k: resolve(v) for k, v in args.items()}
            result = fns[name](**values)
        except Exception as e:                      # the function (or a missing variable) said no: show it
            text = str(e.args[0]) if isinstance(e, KeyError) and e.args else str(e)
            return {"ok": False, "error": type(e).__name__ + (": " + text if text else "")}
        data = json.loads(json.dumps(result, default=_default))
        if save_as:
            VARS[save_as] = data
    return {"ok": True, "result": data, "saved": save_as}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, status, body, ctype="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _local(self):
        return (self.headers.get("Host") or "").rsplit(":", 1)[0] in ("127.0.0.1", "localhost")

    def do_GET(self):
        if not self._local():
            return self._send(403, {"error": "local only"})
        if self.path == "/":
            return self._send(200, PAGE.replace("__TITLE__", html.escape(TITLE)), "text/html")
        if self.path == "/api/functions":
            return self._send(200, describe())
        if self.path == "/api/vars":
            return self._send(200, VARS)
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._local() or self.headers.get("X-UI") != "1":
            return self._send(403, {"error": "forbidden"})
        try:
            size = int(self.headers.get("Content-Length") or 0)
            if size > 1_000_000:
                return self._send(413, {"error": "too big"})
            body = json.loads(self.rfile.read(size) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("a JSON object is needed")
        except (ValueError, json.JSONDecodeError) as e:
            return self._send(400, {"error": "bad request: " + str(e)})
        if self.path == "/api/call":
            save_as = body.get("save_as") or None
            if save_as is not None and not (isinstance(save_as, str) and NAME.match(save_as)):
                return self._send(400, {"error": "a variable name is letters, digits and _"})
            if not isinstance(body.get("args", {}), dict):
                return self._send(400, {"error": "args must be an object"})
            try:
                return self._send(200, call(str(body.get("name")), body.get("args", {}), save_as))
            except LookupError:
                return self._send(404, {"error": "no such function"})
        if self.path == "/api/delete_var":
            VARS.pop(str(body.get("name")), None)
            return self._send(200, VARS)
        self._send(404, {"error": "not found"})


def make_server(host="127.0.0.1", port=8000):
    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    port = int(argv[argv.index("--port") + 1]) if "--port" in argv else 8000
    server = None
    for p in range(port, port + 10):
        try:
            server = make_server("127.0.0.1", p)
            break
        except OSError:
            continue
    if server is None:
        sys.exit("could not find a free port near %d; use --port N" % port)
    url = "http://127.0.0.1:%d" % server.server_address[1]
    print(TITLE + " is running at " + url + "   (Ctrl+C stops it)")
    if "--open" in argv:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
body{font:15px/1.4 system-ui,sans-serif;margin:0;color:#1b1b1b;background:#f6f6f4}
header{background:#1f2933;color:#fff;padding:10px 16px;font-size:18px}
main{display:flex;gap:16px;padding:16px;align-items:flex-start;flex-wrap:wrap}
nav{width:240px;background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px}
nav input{width:100%;box-sizing:border-box;padding:6px;margin-bottom:6px}
nav button{display:block;width:100%;text-align:left;border:0;background:none;padding:5px 6px;cursor:pointer;border-radius:4px;font:inherit}
nav button:hover,nav button.on{background:#e3ecf7}
section{flex:1;min-width:300px;background:#fff;border:1px solid #ddd;border-radius:6px;padding:12px 16px}
h2{margin:0 0 4px;font-size:18px} .doc{color:#555;margin:0 0 10px}
label{display:block;font-weight:600;margin-top:8px} .hint{font-weight:400;color:#777}
textarea{width:100%;box-sizing:border-box;font:13px/1.35 ui-monospace,monospace;padding:6px;min-height:36px}
.row{display:flex;gap:8px;align-items:center;margin-top:10px;flex-wrap:wrap}
button.run{background:#1f6feb;color:#fff;border:0;border-radius:4px;padding:7px 18px;cursor:pointer;font:inherit}
.ex{font-size:12px;padding:3px 8px;cursor:pointer}
pre{background:#f1f3f5;border-radius:4px;padding:8px;white-space:pre-wrap;word-break:break-word;margin:8px 0 0}
pre.ok{border-left:4px solid #2da44e} pre.err{border-left:4px solid #cf222e;color:#8a1621}
aside{width:260px;background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px 12px}
aside h3{margin:0 0 6px;font-size:15px} .var{display:flex;justify-content:space-between;gap:6px;margin:4px 0;font-size:13px}
.var code{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:170px}
.var button{border:0;background:none;color:#cf222e;cursor:pointer}
</style></head>
<body><header>__TITLE__</header>
<main>
<nav><input id="q" placeholder="search functions"><div id="list"></div></nav>
<section id="form"><p class="doc">Pick a function on the left.</p></section>
<aside><h3>Saved variables</h3><div id="vars"><span class="hint">none yet. Save a result, then type $name in any box.</span></div></aside>
</main>
<script>
const $ = (s) => document.querySelector(s);
let FUNCS = [], current = null;
async function api(path, body) {
  const opts = body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json", "X-UI": "1"}, body: JSON.stringify(body)};
  return (await fetch(path, opts)).json();
}
function parseValue(text) { try { return JSON.parse(text.trim()); } catch (e) { return text; } }
function pretty(v) { return JSON.stringify(v, null, 2); }
function el(tag, props, ...kids) {
  const e = Object.assign(document.createElement(tag), props || {});
  kids.forEach((k) => e.append(k));
  return e;
}
function renderList() {
  const q = $("#q").value.toLowerCase(), box = $("#list");
  box.replaceChildren(...FUNCS.filter((f) => f.name.includes(q)).map((f) => {
    const b = el("button", {textContent: f.name, className: current && current.name === f.name ? "on" : ""});
    b.onclick = () => { current = f; renderList(); renderForm(); };
    return b;
  }));
}
function fillExample(ex) {
  for (const p of current.params) {
    const t = document.getElementById("p_" + p.name);
    if (t) t.value = p.name in ex.args ? pretty(ex.args[p.name]) : "";
  }
  $("#out").textContent = ex.raises ? "this example raises " + ex.raises : "expected: " + pretty(ex.expect);
  $("#out").className = "";
}
function renderForm() {
  const f = current, box = $("#form");
  const kids = [el("h2", {textContent: f.name + "(" + f.params.map((p) => p.name).join(", ") + ")"}), el("p", {className: "doc", textContent: f.doc})];
  f.params.forEach((p) => {
    kids.push(el("label", {htmlFor: "p_" + p.name}, p.name, el("span", {className: "hint", textContent: p.default === null ? "" : "  (optional, default " + p.default + ")"})));
    kids.push(el("textarea", {id: "p_" + p.name, rows: 2, placeholder: p.default === null ? "JSON, e.g. 5, \"text\", [1, 2], {\"a\": 1}, or $variable" : "leave empty for the default"}));
  });
  const row = el("div", {className: "row"});
  const run = el("button", {className: "run", textContent: "Run"});
  run.onclick = runIt;
  row.append(run, el("span", {textContent: "save result as $"}), el("input", {id: "save", size: 10, placeholder: "name"}));
  f.examples.forEach((ex, i) => { const b = el("button", {className: "ex", textContent: "example " + (i + 1)}); b.onclick = () => fillExample(ex); row.append(b); });
  kids.push(row, el("pre", {id: "out", textContent: ""}));
  box.replaceChildren(...kids);
  if (f.examples.length) fillExample(f.examples[0]);
}
async function runIt() {
  const args = {};
  for (const p of current.params) {
    const t = document.getElementById("p_" + p.name);
    if (t.value.trim() === "" && p.default !== null) continue;
    args[p.name] = parseValue(t.value);
  }
  const save = $("#save").value.trim().replace(/^\$/, "");
  const r = await api("/api/call", {name: current.name, args, save_as: save || null});
  const out = $("#out");
  if (r.ok) { out.textContent = pretty(r.result) + (r.saved ? "\n\n(saved as $" + r.saved + ")" : ""); out.className = "ok"; }
  else { out.textContent = r.error || "something went wrong"; out.className = "err"; }
  refreshVars();
}
async function refreshVars() {
  const v = await api("/api/vars"), box = $("#vars"), names = Object.keys(v);
  if (!names.length) { box.replaceChildren(el("span", {className: "hint", textContent: "none yet. Save a result, then type $name in any box."})); return; }
  box.replaceChildren(...names.map((n) => {
    const del = el("button", {textContent: "x", title: "delete"});
    del.onclick = async () => { await api("/api/delete_var", {name: n}); refreshVars(); };
    return el("div", {className: "var", title: pretty(v[n])}, el("code", {textContent: "$" + n + " = " + JSON.stringify(v[n])}), del);
  }));
}
$("#q").oninput = renderList;
(async () => { FUNCS = await api("/api/functions"); renderList(); refreshVars(); })();
</script></body></html>
"""

if __name__ == "__main__":
    main()
'''


def _param_names(signature: str) -> list[str]:
    a = ast.parse(f"def {signature}: pass").body[0].args
    return [x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs)]


def ui_examples(plans: list[Plan]) -> dict[str, list[dict]]:
    """function name -> up to 3 of the plan's examples, with the arguments named, to prefill the forms."""
    out: dict[str, list[dict]] = {}
    for plan in plans:
        for step in plan.steps:
            if not step.signature:
                continue
            names = _param_names(step.signature)
            items = []
            for ex in step.examples[:3]:
                args = dict(zip(names, ex.args))
                args.update(ex.kwargs or {})
                items.append({"args": args, **({"raises": ex.raises} if ex.raises else {"expect": ex.expect})})
            out[step.function.partition(":")[2]] = items
    return out


def render_ui(package: str, title: str, examples: dict[str, list[dict]]) -> str:
    """The text of ui.py: a standard-library-only local web page with a form for every public function."""
    values = {"PACKAGE": repr(package), "TITLE_REPR": repr(title), "EXAMPLES": repr(json.dumps(examples)),
              "TITLE_TEXT": re.sub(r"[\r\n\"'\\]+", " ", title).strip() or package}
    return re.sub(r"__(PACKAGE|TITLE_TEXT|TITLE_REPR|EXAMPLES)__", lambda m: values[m.group(1)], UI_TEMPLATE)


def _detect_package(root: Path) -> str:
    found = sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "__init__.py").is_file()
                   and p.name not in ("tests", "venv", ".venv", "node_modules") and not p.name.startswith("."))
    if len(found) != 1:
        raise ProjectError(f"cannot tell which package to make a UI for ({', '.join(found) or 'none found'} in {root}); use --package NAME")
    return found[0]


def write_ui(root: Path, package: str, title: str, plans: list[Plan], *, commit: bool) -> str | None:
    """Write <root>/ui.py and commit it (as the repository's configured identity). Returns the short hash."""
    (root / "ui.py").write_text(render_ui(package, title, ui_examples(plans)), encoding="utf-8")
    if commit and (root / ".git").exists():
        return commit_paths(root, ["ui.py"], "feat: add a small local UI (ui.py)")
    return None


def cmd_ui(args, out, **_):
    root = Path(args.workspace).expanduser()
    if not root.is_dir():
        raise ProjectError(f"{root} is not a folder")
    plans = [load_plan(p) for p in (args.plans or [])]
    package = args.package or (plans[0].steps[0].function.partition(".")[0] if plans else _detect_package(root))
    title = args.title or package.replace("_", " ").title()
    short = write_ui(root, package, title, plans, commit=not args.no_commit)
    out(f"wrote {root / 'ui.py'}" + (f" and committed it ({short})" if short else ""))
    out(f"run it:  cd {root} && python ui.py        then open http://127.0.0.1:8000")
    if args.run:
        return subprocess.call([sys.executable, "ui.py", "--port", str(args.port)], cwd=str(root))
    return EXIT_OK


APP_TEMPLATE = r'''#!/usr/bin/env python3
"""__TITLE_TEXT__: a tiny web app (generated by codeagent). Run:  python app.py   then open http://127.0.0.1:8000

The pages come from the package: render_page(state) returns the HTML, and handle_<action>(state, form) returns the
new state for a form posted to /<action>. The state is kept in data.json next to this file. Only this computer can connect."""
import html
import importlib
import json
import os
import sys
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PACKAGE = __PACKAGE__
TITLE = __TITLE_REPR__
DATA = os.path.join(HERE, "data.json")
LOCK = threading.Lock()
LOCAL = ("127.0.0.1", "localhost")
CSP = "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
pkg = importlib.import_module(PACKAGE)


def load_state():
    try:
        with open(DATA, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        init = getattr(pkg, "initial_state", None)
        return init() if callable(init) else []


def save_state(state):
    text = json.dumps(state, indent=2)                  # fails here, before anything is written, if it is not JSON
    fd, tmp = tempfile.mkstemp(dir=HERE, prefix=".data-")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, DATA)


def actions():
    return sorted(n[len("handle_"):] for n in dir(pkg) if n.startswith("handle_") and callable(getattr(pkg, n)))


def error_page(what, detail):
    return ("<!doctype html><meta charset='utf-8'><title>Error</title><p>" + html.escape(what) + ": <b>" +
            html.escape(detail) + "</b></p><p><a href='/'>Back</a></p>")


def describe(e):
    text = str(e.args[0]) if isinstance(e, KeyError) and e.args else str(e)
    return type(e).__name__ + (": " + text if text else "")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, status, body, headers=None):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _host_ok(self):
        return (self.headers.get("Host") or "").rsplit(":", 1)[0] in LOCAL

    def _origin_ok(self):
        origin = self.headers.get("Origin")
        if origin:
            return urlparse(origin).hostname in LOCAL and urlparse(origin).port == self.server.server_address[1]
        return self.headers.get("Sec-Fetch-Site") in (None, "same-origin", "none")

    def do_GET(self):
        if not self._host_ok():
            return self._send(403, error_page("Refused", "local only"))
        if urlparse(self.path).path != "/":
            return self._send(404, error_page("Not found", self.path))
        try:
            page = pkg.render_page(load_state())
            if not isinstance(page, str):
                raise TypeError("render_page must return the page as text")
        except Exception as e:
            return self._send(500, error_page("The page could not be drawn", describe(e)))
        self._send(200, page)

    def do_POST(self):
        if not self._host_ok() or not self._origin_ok():
            return self._send(403, error_page("Refused", "this request did not come from the page"))
        action = urlparse(self.path).path.strip("/")
        if action not in actions():
            return self._send(404, error_page("Not found", self.path))
        size = int(self.headers.get("Content-Length") or 0)
        if size > 100_000:
            return self._send(413, error_page("Too big", "the form is too large"))
        form = {k: v[0] for k, v in parse_qs(self.rfile.read(size).decode("utf-8", "replace"), keep_blank_values=True).items()}
        with LOCK:
            try:
                new_state = getattr(pkg, "handle_" + action)(load_state(), form)
                if new_state is None:
                    raise TypeError("handle_%s returned nothing; it must return the new state" % action)
                save_state(new_state)
            except Exception as e:
                return self._send(400, error_page("Could not do that", describe(e)))
        self._send(303, "", {"Location": "/"})


def make_server(host="127.0.0.1", port=8000):
    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    port = int(argv[argv.index("--port") + 1]) if "--port" in argv else 8000
    server = None
    for p in range(port, port + 10):
        try:
            server = make_server("127.0.0.1", p)
            break
        except OSError:
            continue
    if server is None:
        sys.exit("could not find a free port near %d; use --port N" % port)
    url = "http://127.0.0.1:%d" % server.server_address[1]
    print(TITLE + " is running at " + url + "   (Ctrl+C stops it)")
    if "--open" in argv:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
'''


def render_app(package: str, title: str) -> str:
    """The text of app.py: a standard-library-only local web app around render_page / handle_<action>."""
    values = {"PACKAGE": repr(package), "TITLE_REPR": repr(title),
              "TITLE_TEXT": re.sub(r"[\r\n\"'\\]+", " ", title).strip() or package}
    return re.sub(r"__(PACKAGE|TITLE_TEXT|TITLE_REPR)__", lambda m: values[m.group(1)], APP_TEMPLATE)


def has_app_functions(root: Path, package: str) -> bool:
    """True when the package defines render_page (the model-built pages that app.py serves)."""
    return any(re.search(r"^def render_page\(", p.read_text(encoding="utf-8", errors="replace"), re.M)
               for p in (Path(root) / package).glob("*.py"))


def write_app(root: Path, package: str, title: str, *, commit: bool) -> str | None:
    if not has_app_functions(root, package):
        raise ProjectError(f"{package} has no render_page(state) function, so there is no page for app.py to serve "
                           f"(see section 13 of docs/PLAN_GUIDE_FOR_AI.md)")
    (root / "app.py").write_text(render_app(package, title), encoding="utf-8")
    if commit and (root / ".git").exists():
        return commit_paths(root, ["app.py"], "feat: add the web app (app.py)")
    return None


def cmd_app(args, out, **_):
    root = Path(args.workspace).expanduser()
    if not root.is_dir():
        raise ProjectError(f"{root} is not a folder")
    package = args.package or _detect_package(root)
    title = args.title or package.replace("_", " ").title()
    short = write_app(root, package, title, commit=not args.no_commit)
    out(f"wrote {root / 'app.py'}" + (f" and committed it ({short})" if short else ""))
    out(f"run it:  cd {root} && python app.py        then open http://127.0.0.1:8000")
    if args.run:
        return subprocess.call([sys.executable, "app.py", "--port", str(args.port)], cwd=str(root))
    return EXIT_OK


# ==========================================================================
# failures and fixes
# A step that cannot be rescued is parked (the run goes on) and written into FAILURES.md: a file that can be pasted
# into ANY AI chat. The AI answers with a "fix pack" (JSON); `codeagent fix` applies it; the next run picks it up.
# ==========================================================================

class FixError(Exception):
    pass


@dataclass
class StepFailure:
    plan_title: str
    plan_path: str
    step: Step
    reason: str
    detail: dict

    @property
    def id(self) -> str:
        return f"{_slug_title(self.step.title)[:40]}-{hashlib.sha1((self.plan_title + self.step.title).encode()).hexdigest()[:6]}"


BLOCKED_PREFIX = "the step description names"


def collect_failures(plan: Plan, plan_path: str | Path, results: list[StepResult],
                     failed_files: set[str] | None = None) -> tuple[list[StepFailure], list[str]]:
    """(failures that need a fix, titles of steps only waiting for a file that a failing step should have made).
    `failed_files` carries the files of failing and waiting steps from one plan to the next. A step whose missing file
    no failing step is meant to create (a misspelt path, a wrong `context`) is a failure of its own, never "waiting"."""
    failed_files = set() if failed_files is None else failed_files
    by_title = {s.title: s for s in plan.steps}
    failures, blocked = [], []
    for r in results:
        if r.status != "failed" or r.title not in by_title:
            continue
        step = by_title[r.title]
        waiting = False
        if r.reason.startswith(BLOCKED_PREFIX):
            m = re.match(rf"{re.escape(BLOCKED_PREFIX)} (.*?), but it is missing", r.reason)
            names = [n.strip() for n in m.group(1).split(", ")] if m else []
            waiting = any(n in failed_files for n in names)
        if waiting:
            blocked.append(r.title)
        else:
            failures.append(StepFailure(plan.title, str(plan_path), step, r.reason, r.detail))
        failed_files.update(step.files)
    return failures, blocked


FIX_PACK_FORMAT = '''\
```json
{
  "fixes": [
    {
      "failure": "<the failure id from this file>",
      "plan_step": {
        "description": "the rewritten recipe (code-shaped lines, see the rules above)",
        "examples": [ {"args": [1], "expect": 2} ]
      },
      "body": "optional: the exact function body, if you are sure of it",
      "lessons": ["optional: one short rule the model must follow in every later step"],
      "scope": "project"
    }
  ]
}
```
Only these keys are allowed. `plan_step` may contain `description`, `examples`, `signature`, `context`, `commit` (never `title`);
include only what changes. `body` is the function body only (no `def` line). `scope` is `project` (default) or `global`
(the lesson then applies to every project). Reply with the JSON object and nothing else.'''

RECIPE_RULES = '''\
- Write the algorithm as a recipe of tiny statements: a numbered list, every variable named, real Python expressions.
- One statement per line. A loop is its own line (`For each item in cart:`), its statements are `-` bullets below it.
  Never put a `for` and an `if` on one line, never join two statements with `and` or `;`.
- Write conditions as code, not English: `2. if len(text) <= limit: return text`.
- Never reassign a parameter (`category = category.strip()`); use a new name (`cleaned = category.strip()`).
- Assign the result of a helper call even if only its error matters: `found = find_transaction(items, id)`.
- A standard-library import is its own first line: `1. import re`.
- At most 6 numbered lines and one loop; split a bigger function into two steps.
- Check every example by hand: the examples are the tests, a wrong example makes the step impossible.
- For ties ("first wins") prefer a loop with a strict `>` over `sorted(..., key=lambda ...)` with a negated key.'''


def render_failure_report(project: str, failures: list[StepFailure], blocked: list[str]) -> str:
    out = [f"# Help the local code agent: {len(failures)} failing step{'s' if len(failures) != 1 else ''} in {project}", "",
           "Paste this whole file into an AI chat (any AI) and ask it to fix the steps. It must answer with ONE JSON object,",
           "a *fix pack*, in the format at the bottom. Then, in the terminal:", "",
           "    codeagent fix fix.json            # or:  codeagent fix -   (paste the JSON, then press Ctrl-D)",
           "    codeagent build project.json      # the same command as before; it continues where it stopped", "",
           "## How the agent works (so the fix fits it)", "",
           "- A small local model (about 0.8B parameters) writes one function body per step. The agent writes the tests from",
           "  the step's `examples`: the examples are the only specification, so they must be right.",
           "- The model sees the step `description`, the `def` line, the examples, the *signatures* of helper functions and",
           "  some rules learned from earlier fixes. It copies recipe lines literally, and often drops or rewrites a line it",
           "  thinks is optional. It had several normal attempts, the recipe itself as code, and a repair prompt.", "",
           "## Rules for a description it can follow", "", RECIPE_RULES, "", "## Failing steps", ""]
    for i, f in enumerate(failures, 1):
        d = f.detail or {}
        step_json = json.dumps({k: v for k, v in asdict(f.step).items() if v not in (None, [], "")}, indent=2, ensure_ascii=False)
        out += [f"### {i}. {f.step.title}", "", f"- failure id: `{f.id}`", f"- plan: `{f.plan_path}` (plan title: {f.plan_title})",
                f"- why it stopped: {f.reason}", "", "The step as it is in the plan:", "", "```json", step_json, "```", ""]
        bodies = d.get("bodies") or []
        if bodies:
            out += ["What the model wrote in its last attempts (oldest first):", ""]
            for n, body in enumerate(bodies, 1):
                out += [f"attempt {n}:", "```python", body.strip("\n")[:1500], "```", ""]
        if d.get("output"):
            out += ["What the tests reported for the last attempt:", "", "```", "\n".join(d["output"].strip().splitlines()[-25:])[:2500], "```", ""]
        if d.get("notes"):
            out += ["What the agent noticed:", ""] + [f"- {n}" for n in d["notes"]] + [""]
        note = _missing_lines_note(f.step.description, bodies[-1] if bodies else "")
        if note:
            out += ["Recipe lines missing from the last attempt:", "", note.split(":\n", 1)[-1], ""]
    if blocked:
        out += ["## Waiting (no fix needed)", "", "These steps only wait for a file that a failing step above should create:", ""]
        out += [f"- {t}" for t in blocked] + [""]
    out += ["## The fix pack", "", "Fix the steps above, preferably by rewriting `plan_step.description` (and `examples` if one of them is",
            "wrong), keeping the step's title. Add a `lesson` when the same mistake could happen in other steps.", "",
            FIX_PACK_FORMAT, ""]
    return "\n".join(out)


def write_failure_reports(root: str | Path, project: str, failures: list[StepFailure], blocked: list[str]) -> Path | None:
    """Writes .codeagent/FAILURES.md and the id registry `codeagent fix` uses; removes both when nothing failed."""
    root = Path(root)
    report, index = root / DIR / "FAILURES.md", root / DIR / "failures.json"
    if not failures:
        for p in (report, index):
            p.unlink(missing_ok=True)
        return None
    (root / DIR).mkdir(exist_ok=True)
    report.write_text(render_failure_report(project, failures, blocked), encoding="utf-8")
    index.write_text(json.dumps({f.id: {"plan_file": f.plan_path, "step_title": f.step.title, "plan_title": f.plan_title}
                                 for f in failures}, indent=2), encoding="utf-8")
    return report


def extract_json(text: str) -> object:
    """The JSON in an AI's reply: bare, or inside a ``` fence, or with words around it."""
    text = text.strip()
    candidates = [text] + re.findall(r"```(?:json)?\s*(.*?)```", text, re.S)
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start:end + 1])
    for c in candidates:
        try:
            return json.loads(c.strip())
        except json.JSONDecodeError:
            continue
    raise FixError("that is not JSON. Paste the fix pack exactly as the AI wrote it (the { ... } part).")


PLAN_STEP_KEYS = {"description", "examples", "signature", "context", "commit"}
FIX_KEYS = {"failure", "plan_step", "body", "lessons", "scope"}


def apply_fix_pack(root: str | Path, text: str, *, now: Callable[[], str] = lambda: time.strftime("%Y%m%d-%H%M%S")) -> list[str]:
    """Validate everything first, then apply: plan edits (with a backup), body overrides, lessons."""
    root = Path(root)
    data = extract_json(text)
    fixes = data.get("fixes") if isinstance(data, dict) and "fixes" in data else [data]
    if not isinstance(fixes, list) or not fixes or not all(isinstance(f, dict) for f in fixes):
        raise FixError("the fix pack must be a JSON object with a \"fixes\" list")
    try:
        registry = json.loads((root / DIR / "failures.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        registry = {}
    plan_edits: dict[str, dict] = {}                      # plan file -> parsed json (all edits applied in memory)
    actions: list[Callable[[], str]] = []
    messages: list[str] = []
    for n, fix in enumerate(fixes, 1):
        label = f"fix {n}"
        unknown = set(fix) - FIX_KEYS
        if unknown:
            raise FixError(f"{label}: unknown key(s) {sorted(unknown)}; allowed: {sorted(FIX_KEYS)}")
        fid = fix.get("failure")
        patch = fix.get("plan_step")
        entry = registry.get(fid) if isinstance(fid, str) else None
        if fid and entry is None:
            raise FixError(f"{label}: unknown failure id {fid!r}; the ids are in {root / DIR / 'FAILURES.md'}")
        if entry is None:
            title = patch.get("title") if isinstance(patch, dict) else None
            matches = [e for e in registry.values() if e["step_title"] == title]
            if len(matches) != 1:
                raise FixError(f"{label}: say which step it is for with \"failure\": \"<id>\" (see FAILURES.md)")
            entry = matches[0]
        step_title = entry["step_title"]
        if patch is not None:
            if not isinstance(patch, dict) or set(patch) - PLAN_STEP_KEYS - {"title"}:
                raise FixError(f"{label}: plan_step may only contain {sorted(PLAN_STEP_KEYS)}")
            if "title" in patch and patch["title"] != step_title:
                raise FixError(f"{label}: do not rename the step (it is {step_title!r}); leave `title` out")
            plan_file = entry["plan_file"]
            if plan_file not in plan_edits:
                try:
                    plan_edits[plan_file] = json.loads(Path(plan_file).read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as e:
                    raise FixError(f"{label}: cannot read the plan {plan_file}: {e}") from e
            steps = [s for s in plan_edits[plan_file].get("steps", []) if s.get("title") == step_title]
            if len(steps) != 1:
                raise FixError(f"{label}: the step {step_title!r} is not in {plan_file}")
            steps[0].update({k: v for k, v in patch.items() if k != "title"})
            messages.append(f"{label}: rewrote {', '.join(sorted(k for k in patch if k != 'title'))} of {step_title!r} in {plan_file}")
        body = fix.get("body")
        if body is not None:
            if not isinstance(body, str) or not body.strip():
                raise FixError(f"{label}: body must be the function body as text")
            try:
                ast.parse("def _f():\n" + textwrap.indent(textwrap.dedent(body), "    "))
            except SyntaxError as e:
                raise FixError(f"{label}: the body is not valid Python ({e.msg}, line {e.lineno})") from e
            target = root / DIR / "overrides" / f"{_slug_title(step_title)}.py"
            actions.append(lambda target=target, body=body, t=step_title: (
                target.parent.mkdir(parents=True, exist_ok=True),
                target.write_text(textwrap.dedent(body).strip("\n") + "\n", encoding="utf-8"),
                f"saved the body for {t!r}; the next run tries it first")[2])
        lessons = fix.get("lessons", [])
        if lessons:
            if not isinstance(lessons, list) or not all(isinstance(l, str) and l.strip() for l in lessons) or len(lessons) > 5:
                raise FixError(f"{label}: lessons must be up to 5 short texts")
            if any(len(l) > 240 for l in lessons):
                raise FixError(f"{label}: a lesson is longer than 240 characters; make it one short rule")
            scope = fix.get("scope", "project")
            if scope not in ("project", "global"):
                raise FixError(f"{label}: scope must be \"project\" or \"global\"")
            lessons_file = global_lessons_path() if scope == "global" else root / DIR / "lessons.json"
            actions.append(lambda lessons=lessons, f=lessons_file, sc=scope: _add_lessons(f, lessons, sc))
        if patch is None and body is None and not lessons:
            raise FixError(f"{label}: it changes nothing (needs plan_step, body or lessons)")
    for plan_file, raw in plan_edits.items():                  # the edited plans must still be valid plans
        try:
            plan = parse_plan(json.dumps(raw))
        except PlanError as e:
            raise FixError(f"the edited plan {plan_file} would be invalid: {e}") from e
        for st in plan.steps:
            messages += [f"warning: {st.title}: {w}" for w in lint_step(st)]
        actions.append(lambda plan_file=plan_file, raw=raw: _write_plan_with_backup(Path(plan_file), raw, now()))
    return messages + [a() for a in actions]


def _write_plan_with_backup(path: Path, raw: dict, stamp: str) -> str:
    backup = path.with_name(f"{path.name}.bak-{stamp}")
    shutil.copy2(path, backup)
    path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return f"saved {path} (the old one is {backup.name})"


def _add_lessons(path: Path, lessons: list[str], scope: str) -> str:
    try:
        have = json.loads(path.read_text(encoding="utf-8")).get("lessons", [])
    except (OSError, json.JSONDecodeError):
        have = []
    new = [l.strip() for l in lessons if l.strip() not in have]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"lessons": have + new}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return f"added {len(new)} {scope} lesson{'s' if len(new) != 1 else ''} to {path}"


def cmd_fix(args, out, **_):
    text = sys.stdin.read() if args.fix == "-" else Path(args.fix).read_text(encoding="utf-8")
    root = _fix_workspace(args)
    if not (root / DIR / "failures.json").is_file():
        raise FixError(f"nothing to fix in {root}: no failed steps are recorded there. If you built with --workspace, "
                       f"pass the same --workspace here.")
    for line in apply_fix_pack(root, text):
        out(line)
    out("done. Run the same `codeagent build ...` command again: it continues with the fixed steps.")
    return EXIT_OK


def _fix_workspace(args) -> Path:
    if args.workspace:
        return Path(args.workspace).expanduser()
    project_file = Path(args.project) if args.project else Path("project.json")
    if not project_file.is_file():
        raise FixError("say where the project is: --project project.json (or --workspace DIR); no project.json here")
    return load_project(project_file).workspace


# ==========================================================================
# project: one command from plans to a published repository
#   codeagent build project.json
# creates a PRIVATE repo in your account, builds every plan (rescuing and parking failing steps instead of
# stopping), consolidates the modules, and publishes - committing as you, with the token from your .env.
# ==========================================================================

class ProjectError(Exception):
    pass


PROJECT_KEYS = {"name", "description", "plans", "groups", "workspace"}


@dataclass
class Project:
    name: str
    repo: str
    description: str
    plans: list[Path]
    groups: Path | None
    workspace: Path
    file: Path


def load_project(path: str | Path) -> Project:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ProjectError(f"cannot read {path}: {e}") from e
    except json.JSONDecodeError as e:
        raise ProjectError(f"{path} is not valid JSON ({e.msg}, line {e.lineno})") from e
    if not isinstance(raw, dict) or set(raw) - PROJECT_KEYS:
        raise ProjectError(f"{path}: allowed keys are {sorted(PROJECT_KEYS)}" + (f"; unknown: {sorted(set(raw) - PROJECT_KEYS)}" if isinstance(raw, dict) else ""))
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ProjectError(f"{path}: \"name\" is required (it becomes the repository name, e.g. \"expense-manager\")")
    plans = raw.get("plans")
    if not isinstance(plans, list) or not plans or not all(isinstance(p, str) for p in plans):
        raise ProjectError(f"{path}: \"plans\" must list the plan files in the order they are built")
    base = path.parent
    resolved = [(base / p).expanduser() for p in plans]
    missing = [str(p) for p in resolved if not p.is_file()]
    if missing:
        raise ProjectError(f"{path}: plan file not found: {', '.join(missing)}")
    groups = (base / raw["groups"]).expanduser() if raw.get("groups") else None
    if groups and not groups.is_file():
        raise ProjectError(f"{path}: groups file not found: {groups}")
    repo = repo_name(name)
    workspace = Path(raw.get("workspace") or Path("~/agent-projects") / repo).expanduser()
    return Project(name.strip(), repo, str(raw.get("description") or "").strip(), resolved, groups, workspace, path)


def _project_state(root: Path) -> dict:
    try:
        return json.loads((root / DIR / "project.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_project_state(root: Path, state: dict) -> None:
    (root / DIR).mkdir(exist_ok=True)
    (root / DIR / "project.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def _readme(project: Project) -> str:
    return (f"# {project.name}\n\n{project.description + chr(10) * 2 if project.description else ''}"
            "Built step by step by [codeagent](https://github.com/Yad4o/local-code-agent): a small local model writes "
            "one function per step, every step is tested and committed on its own.\n")


def _init_project_repo(ws: Workspace, project: Project, identity: tuple[str, str] | None) -> None:
    ws.root.mkdir(parents=True, exist_ok=True)
    ensure_repo(ws.root)
    if identity:
        set_identity(ws.root, *identity)
    if _run(ws.root, "rev-parse", "--verify", "-q", "HEAD", check=False):
        return
    (ws.root / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    if not (ws.root / "README.md").exists():
        (ws.root / "README.md").write_text(_readme(project), encoding="utf-8")
    commit_paths(ws.root, [".gitignore", "README.md"], f"chore: start {project.name}")


def _publish(ws: Workspace, secrets: Secrets, state: dict, branch: str, out: Callable[[str], None], *,
             main: bool) -> bool:
    """Fast-forward main to the work branch (when asked) and push to GitHub (when there is a repo). Network trouble
    is a warning: the work is safe locally and the next `build` pushes it. A secret anywhere in the history stops
    the push for good. Returns True when everything was pushed."""
    owner, repo = state.get("owner"), state.get("repo")
    pushing = bool(owner and repo and secrets.token)
    url = remote_url(secrets, owner, repo) if pushing else ""
    if pushing:
        try:
            preflight_push(ws.root, (secrets.token,))
        except GitError as e:
            out(f"NOT PUSHED: {e}")
            pushing = False
    try:
        if pushing:
            push_ref(ws.root, url, branch, secrets, owner)
        if main:
            _run(ws.root, "checkout", "-q", "main")
            _run(ws.root, "merge", "--ff-only", "-q", branch)
            if pushing:
                push_ref(ws.root, url, "main", secrets, owner)
    except GitError as e:
        out(f"warning: {secrets.scrub(str(e))}. Your work is safe here; run the same command again to retry.")
        return False
    return pushing


def cmd_build(args, out, *, llm_factory, guard_factory, **_):
    cfg = load_config()
    cfg.stop_on_failure = False                       # a failing step is parked, the run goes on
    project = load_project(args.project)
    ws_path = Path(args.workspace).expanduser() if args.workspace else project.workspace
    secrets = load_secrets(args.env, project.file.parent)
    plans = [(p, load_plan(p)) for p in project.plans]
    titles = [pl.title for _, pl in plans]
    if len(set(titles)) != len(titles):
        raise ProjectError("two plans have the same title; every plan needs its own title")
    for _, plan in plans:
        _check_test_command(cfg, plan)
    package, groups = None, None
    if project.groups:
        package, groups = load_groups(project.groups)
        created = {st.function.partition(":")[0].split(".")[-1] for _, pl in plans for st in pl.steps}
        unknown = sorted(m for olds in groups.values() for m in olds if m not in created)
        if unknown:
            raise ConsolidateError(f"{project.groups} lists {', '.join(unknown)}, which no plan creates")
    branch = f"agent/{project.repo}"
    if args.dry_run:
        out(f"[dry run] would build '{project.name}' in {ws_path} on branch {branch}, plans in this order:")
        for _, pl in plans:
            out(f"  {pl.title}: {len(pl.steps)} steps")
        out("  then consolidate" + (f" with {project.groups}" if groups else " (one module per plan)")
            + (", create a private GitHub repo and push" if not args.no_github else ", no GitHub (--no-github)"))
        return EXIT_OK
    for w in secrets.warnings:
        out(f"note: {w}")
    gh, identity = None, None
    if not args.no_github:
        if not secrets.token:
            out("No GitHub token found. Run:  codeagent setup   (it asks for your token once and saves it in ~/.codeagent.env)\n"
                "or put GITHUB_TOKEN=... in a .env file next to project.json. To build without GitHub: add --no-github")
            return EXIT_USAGE
        gh = GitHub(secrets.token, secrets.get("GITHUB_API_URL") or "https://api.github.com")
        try:
            me = gh.whoami()
        except GitHubError as e:
            out(f"error: {secrets.scrub(str(e))}")
            return EXIT_USAGE
        identity = (secrets.get("GIT_AUTHOR_NAME") or me["name"], secrets.get("GIT_AUTHOR_EMAIL") or me["email"])
        out(f"GitHub: {me['login']}   commits will be made as {identity[0]} <{identity[1]}>")
    elif secrets.get("GIT_AUTHOR_NAME") and secrets.get("GIT_AUTHOR_EMAIL"):
        identity = (secrets.get("GIT_AUTHOR_NAME"), secrets.get("GIT_AUTHOR_EMAIL"))
    if ws_path.exists() and any(ws_path.iterdir()) and not (ws_path / DIR / "project.json").exists():
        out(f"error: {ws_path} already has files in it and was not made by `codeagent build`. Use another --workspace.")
        return EXIT_USAGE
    ws = Workspace(ws_path, cfg.max_file_bytes)
    log, close = _make_log(out, ws.root / DIR / "agent.log")
    stopper = _Stopper()
    stopper.install()
    try:
        try:
            _init_project_repo(ws, project, identity)
        except GitError as e:
            out(f"error: {e}")
            return EXIT_USAGE
        state = _project_state(ws.root)
        if gh and not state.get("repo"):
            try:
                data = gh.create_unique_repo(project.repo, project.description or project.name)
            except GitHubError as e:
                out(f"error: {secrets.scrub(str(e))}")
                return EXIT_USAGE
            owner, _, repo = data["full_name"].partition("/")
            state = {"owner": owner, "repo": repo, "url": data.get("html_url", f"https://github.com/{owner}/{repo}"),
                     "main_pushed": False}
            _save_project_state(ws.root, state)
            out(f"created the private repository {owner}/{repo}" + ("" if repo == project.repo else f"  (the name {project.repo} was taken)"))
        elif not state:
            _save_project_state(ws.root, {"local": True})
        if state.get("repo"):
            url = remote_url(secrets, state["owner"], state["repo"])
            if "origin" in _run(ws.root, "remote", check=False).split():
                _run(ws.root, "remote", "set-url", "origin", url)
            else:
                _run(ws.root, "remote", "add", "origin", url)
            if not state.get("main_pushed"):
                try:
                    preflight_push(ws.root, (secrets.token or "",))
                    push_ref(ws.root, url, "main", secrets, state["owner"])
                    state["main_pushed"] = True
                    _save_project_state(ws.root, state)
                except GitError as e:
                    out(f"warning: could not push main yet ({secrets.scrub(str(e))})")
        if (ws.root / DIR / MARKER).exists():
            out("this project is already built and consolidated.")
            _publish(ws, secrets, state, branch, out, main=not args.no_main)
            return EXIT_OK
        llm = llm_factory(cfg)
        failures: list[StepFailure] = []
        blocked: list[str] = []
        rescued: list[str] = []
        failed_files: set[str] = set()
        for plan_path, plan in plans:
            for st in plan.steps:
                for w in lint_step(st):
                    log(f"note: {plan.title} / {st.title}: {w}")
            runner = Runner(cfg, plan, ws, llm, guard=guard_factory(cfg, ws.root) if guard_factory else None, log=log,
                            should_stop=stopper, live=LivePrinter() if args.live else None, branch=branch, plan_path=plan_path)
            report = runner.run()
            out(report.summary())
            if report.outcome == "paused":
                out("paused. Run the same command again to continue.")
                return EXIT_PAUSED
            rescued += [f"{r.title} ({r.via})" for r in report.results if r.status == "done" and r.via]
            f, b = collect_failures(plan, plan_path, report.results, failed_files)
            failures += f
            blocked += b
            _publish(ws, secrets, state, branch, out, main=False)
        report_file = write_failure_reports(ws.root, project.name, failures, blocked)
        if failures:
            out("")
            out(f"{len(failures)} step{'s' if len(failures) != 1 else ''} could not be built even with the repair tries"
                + (f" (and {len(blocked)} waiting for them)" if blocked else "") + ". The rest of the project is built and saved.")
            for f in failures:
                out(f"  - {f.step.title}")
            out(f"\nTo fix it:\n  1. Open {report_file}\n  2. Paste the whole file into any AI chat; it replies with a fix pack (JSON)\n"
                f"  3. Save the reply as fix.json and run:  codeagent fix fix.json --workspace {ws.root}\n"
                f"  4. Run again:  codeagent build {args.project}" + (f" --workspace {ws.root}" if args.workspace else ""))
            return EXIT_FAILED
        if not args.no_consolidate:
            try:
                use_groups = groups
                if use_groups is None:
                    suggested = suggest_groups([pl for _, pl in plans])
                    package, use_groups = suggested["package"], suggested["modules"]
                consolidate_workspace(ws, package, use_groups, tidy=True, commit=True, out=out)
            except ConsolidateError as e:
                out(f"warning: the modules were not merged ({e}). The project itself is built; fix the groups and run: "
                    f"codeagent consolidate GROUPS.json --workspace {ws.root}")
        if not args.no_ui:
            try:
                ui_package = package or suggest_groups([pl for _, pl in plans])["package"]
                write_ui(ws.root, ui_package, project.name, [pl for _, pl in plans], commit=True)
                out("wrote ui.py (a small local web UI for every function)")
            except (ConsolidateError, GitError, ProjectError, OSError) as e:
                out(f"warning: ui.py was not made ({e})")
        if not args.no_ui:
            try:
                app_package = package or suggest_groups([pl for _, pl in plans])["package"]
                if has_app_functions(ws.root, app_package):
                    write_app(ws.root, app_package, project.name, commit=True)
                    out("wrote app.py (a web app: the pages and actions your plans built)")
            except (ConsolidateError, GitError, ProjectError, OSError) as e:
                out(f"warning: app.py was not made ({e})")
        published = _publish(ws, secrets, state, branch, out, main=not args.no_main)
        out("")
        out(f"DONE: {sum(len(pl.steps) for _, pl in plans)} steps built in {ws.root}")
        if (ws.root / "app.py").exists():
            out(f"      the app:  cd {ws.root} && python app.py     then open http://127.0.0.1:8000")
        if (ws.root / "ui.py").exists():
            out(f"      try it:  cd {ws.root} && python ui.py     then open http://127.0.0.1:8000")
        if rescued:
            out(f"      {len(rescued)} needed help: " + "; ".join(rescued))
        if state.get("url") and published:
            out(f"      {state['url']}   (private, branch main)")
        elif state.get("url"):
            out(f"      {state['url']}   (not pushed yet: run the same command again)")
        return EXIT_OK
    except WorkspaceBusy as e:
        out(str(e))
        return EXIT_BUSY
    except KeyboardInterrupt:
        out("aborted")
        return 130
    finally:
        stopper.restore()
        close()


TOKEN_SHAPE = re.compile(r"^(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})$")


def clean_pasted_token(raw: str) -> str:
    """What a terminal paste can add around a token: bracketed-paste escape codes, quotes, spaces, a `GITHUB_TOKEN=`
    or `export ` in front."""
    t = re.sub(r"\x1b\[[0-9;?]*[A-Za-z~]", "", raw).replace("^[[200~", "").replace("^[[201~", "").strip()
    t = re.sub(r"^(?:export\s+)?(?:GITHUB_TOKEN|GH_TOKEN)\s*=\s*", "", t)
    return t.strip().strip("\"'").strip()


def cmd_setup(args, out, **_):
    """Ask for the GitHub token once, check it, and save it in a private file (mode 600)."""
    path = Path(args.env or Path.home() / ".codeagent.env").expanduser()
    if args.visible:
        raw = input("Paste your GitHub token (it will be visible): ")
    elif sys.stdin.isatty():
        raw = getpass.getpass("Paste your GitHub token (nothing is shown while you type, just paste and press Enter): ")
    else:
        raw = sys.stdin.readline()
    token = clean_pasted_token(raw)
    if not token:
        out("no token received. Nothing was saved. Try again, or use: codeagent setup --visible")
        return EXIT_USAGE
    out(f"received {len(token)} character{'s' if len(token) != 1 else ''}, starting with {token[:4]!r}")
    if len(token) < 20:
        out("That is far too short to be a GitHub token (they are about 40 characters and start with ghp_). "
            "The paste did not reach the program. Nothing was saved. Try:\n"
            "  - codeagent setup --visible        (you can see what arrives)\n"
            "  - paste with Ctrl+Shift+V (not Ctrl+V) in the terminal, or right-click > Paste\n"
            "  - or save it by hand:  nano ~/.codeagent.env   with one line  GITHUB_TOKEN=ghp_...  then  chmod 600 ~/.codeagent.env")
        return EXIT_USAGE
    if not TOKEN_SHAPE.match(token):
        out("note: this does not look like a GitHub token (they start with ghp_ or github_pat_); checking it anyway")
    api = (os.environ.get("GITHUB_API_URL") or "https://api.github.com")
    try:
        me = GitHub(token, api).whoami()
    except GitHubError as e:
        out(f"error: {e}")
        return EXIT_USAGE
    values = parse_dotenv(path.read_text(encoding="utf-8")) if path.is_file() else {}
    values["GITHUB_TOKEN"] = token
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")
    path.chmod(0o600)
    out(f"saved to {path} (only you can read it)")
    out(f"signed in as {me['login']}; commits will be made as {me['name']} <{me['email']}>")
    if me["scopes"] and "repo" not in me["scopes"]:
        out("warning: this token does not have the `repo` scope, so creating private repositories will fail. "
            "Make a classic token with `repo` ticked.")
    return EXIT_OK


def cmd_whoami(args, out, **_):
    secrets = load_secrets(args.env, None)
    if not secrets.token:
        out("no token found: run  codeagent setup")
        return EXIT_USAGE
    try:
        me = GitHub(secrets.token, secrets.get("GITHUB_API_URL") or "https://api.github.com").whoami()
    except GitHubError as e:
        out(f"error: {secrets.scrub(str(e))}")
        return EXIT_USAGE
    out(f"signed in as {me['login']}")
    out(f"commits are made as {secrets.get('GIT_AUTHOR_NAME') or me['name']} <{secrets.get('GIT_AUTHOR_EMAIL') or me['email']}>")
    out("token scopes: " + (", ".join(me["scopes"]) if me["scopes"] else "(not shown; a fine-grained token)"))
    for f in secrets.files:
        out(f"read: {f}")
    for w in secrets.warnings:
        out(f"note: {w}")
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

    if (ws_path / DIR / MARKER).exists():
        out(f"{ws_path} was consolidated into fewer modules, so the plans' files no longer exist there and a rerun would "
            f"rebuild everything. Use a new workspace (or delete {ws_path / DIR / MARKER} if you really want to rebuild here).")
        return EXIT_USAGE
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
    sp = sub.add_parser("consolidate", help="merge the one-module-per-function files of a finished project into a few modules")
    sp.add_argument("groups", nargs="?", help="groups.json: which old modules go into which new module")
    sp.add_argument("--suggest", nargs="+", metavar="PLAN", help="print a starter groups.json (one module per plan) and exit")
    sp.add_argument("--workspace"); sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--no-tidy", action="store_true", help="keep unused `x = f(...)` assignments as they are")
    sp.add_argument("--no-commit", action="store_true", help="change the files but do not commit")
    sp = sub.add_parser("build", help="build a whole project: every plan, then merge the modules and publish to a private GitHub repo")
    sp.add_argument("project", help="project.json")
    sp.add_argument("--workspace"); sp.add_argument("--env", help="a .env file with GITHUB_TOKEN (default: see README)")
    sp.add_argument("--dry-run", action="store_true"); sp.add_argument("--live", action="store_true")
    sp.add_argument("--no-github", action="store_true", help="build and commit locally only")
    sp.add_argument("--no-consolidate", action="store_true", help="keep one module per function")
    sp.add_argument("--no-main", action="store_true", help="push the work branch only; leave main alone")
    sp.add_argument("--no-ui", action="store_true", help="do not write ui.py")
    sp = sub.add_parser("ui", help="write ui.py, a small local web UI with a form for every function of a built project")
    sp.add_argument("workspace"); sp.add_argument("--plans", nargs="*", help="plan files, to prefill the forms with their examples")
    sp.add_argument("--package"); sp.add_argument("--title"); sp.add_argument("--no-commit", action="store_true")
    sp.add_argument("--run", action="store_true", help="start it right away"); sp.add_argument("--port", type=int, default=8000)
    sp = sub.add_parser("app", help="write app.py, a web app that serves the render_page / handle_<action> functions of a built project")
    sp.add_argument("workspace"); sp.add_argument("--package"); sp.add_argument("--title"); sp.add_argument("--no-commit", action="store_true")
    sp.add_argument("--run", action="store_true", help="start it right away"); sp.add_argument("--port", type=int, default=8000)
    sp = sub.add_parser("fix", help="apply a fix pack (JSON from an AI chat) to the failing steps")
    sp.add_argument("fix", help="fix.json, or - to read it from the terminal (paste, then Ctrl-D)")
    sp.add_argument("--project", help="project.json (default: ./project.json)"); sp.add_argument("--workspace")
    sp = sub.add_parser("setup", help="save your GitHub token once (asks for it, checks it, saves it in ~/.codeagent.env)")
    sp.add_argument("--env", help="where to save it (default: ~/.codeagent.env)")
    sp.add_argument("--visible", action="store_true", help="show the token while you paste it (if the hidden prompt does not work)")
    sp = sub.add_parser("whoami", help="show which GitHub account and commit identity are used")
    sp.add_argument("--env")
    sp = sub.add_parser("run", help="build a plan")
    sp.add_argument("plan"); sp.add_argument("--workspace"); sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--live", action="store_true",
                    help="show each prompt and the model's raw output (thinking too) as it streams; terminal only, not saved")
    return p


COMMANDS = {"template": cmd_template, "check": cmd_check, "doctor": cmd_doctor,
            "smoke": cmd_smoke, "run": cmd_run, "consolidate": cmd_consolidate, "build": cmd_build, "fix": cmd_fix,
            "setup": cmd_setup, "whoami": cmd_whoami, "ui": cmd_ui, "app": cmd_app}


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
    except (ConfigError, PlanError, ConsolidateError, GitError, WorkspaceError, ProjectError, FixError, GitHubError) as e:
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
