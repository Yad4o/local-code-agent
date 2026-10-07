# local-code-agent

An **offline, resource-safe coding agent** for small local models (built for a
laptop with a 2 GB GPU running Ollama + Qwen 0.8B Q4). You write a **structured
plan**; the agent builds it step by step, the way you pair with an AI: **tests
first, implement, run the tests, commit.** No internet, no dependencies (Python
standard library only).

## Why it is built this way

A 0.8B model cannot "figure out" a project. So the **harness does the thinking
and the model only does tiny, well-defined jobs**:

| The harness (code) decides | The model only answers |
|---|---|
| what step is next, which files may change | "write these exact test files" |
| runs the tests, reads the real output | "write these exact implementation files" |
| retries (bounded), rolls back, commits | "here is the failing output, fix it" |

The model's reply uses one rigid format (`FILE: path` + a fenced code block),
not JSON/tool-calls, which small models follow far more reliably.

```
for each plan step:
  1. model writes TEST files          -> must FAIL (vacuous tests rejected)
  2. model writes IMPLEMENTATION      -> run tests -> failing? feed real output back, retry (max N)
  3. tests green -> git commit exactly this step's files, save progress
  any failure -> every file of the step is rolled back; nothing half-written is left behind
```

## Quick start (Ubuntu)

```bash
git clone <this repo> && cd local-code-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q                 # all tests should pass

ollama list                         # note your model's exact name
export CODEAGENT_MODEL="<name from ollama list>"   # or set `model = ...` in codeagent.toml

codeagent doctor                    # checks Ollama, the model, git, RAM/battery/temp/disk
codeagent smoke --tries 5           # can YOUR model follow the output format? (needs >= 3/5)
codeagent template --out plan.toml  # a plan skeleton (see examples/slugify.toml)
codeagent check plan.toml           # validate a plan without running anything
codeagent run plan.toml --workspace ~/agent-projects/demo
```

Exit codes of `run`: `0` done, `1` a step failed, `2` bad plan/config, `3` paused
(model server down or system resources low; just run it again, finished steps are skipped).

## Plans (always the same structure)

Plans are TOML, validated strictly before anything runs (all mistakes are reported
at once). Every step names **exactly** the files it may touch:

```toml
title = "Slugify utility"
test_command = "python -m pytest -q"

[[step]]
title = "slugify lowercases and joins words with dashes"
description = "Create slugify(text) in slugify/core.py ... 'Hello, World!' -> 'hello-world'. Tests import: from slugify.core import slugify"
commit = "feat: add slugify core function"
tests = ["tests/test_core.py"]                    # written FIRST
files = ["slugify/__init__.py", "slugify/core.py"]  # the only implementation files it may write
```

Tips for a tiny model: **one function per step**, exact input -> output examples,
and name the exact import the tests must use. The agent works on a branch
`agent/<plan-title>`, never on your `main`.

## 24/7 mode

```bash
codeagent daemon --inbox ~/agent-inbox        # drop plan .toml files into the inbox (each needs `workspace = "..."`)
```

`inbox/` holds new plans, `running/` the current one, `done/` and `failed/` the
results with a `*.report.txt` / `*.error.txt` explaining what happened. After a
crash or reboot, interrupted plans are re-queued and resume where they stopped.
Repeated model outages back off (up to 10 min) instead of spinning.

To start it at boot and restart it after crashes, use the systemd user service in
`deploy/codeagent.service` (instructions at the top of that file).

## Safety and "don't crash my laptop"

| Risk | Protection |
|---|---|
| model writes outside the project | all file access is confined to the workspace; `..`, absolute paths, symlink escapes, `.git/` are refused; only files **declared in the step** can be written |
| model runs dangerous commands | no shell; only allowlisted programs (`python`, `pytest` by default); `python` means the agent's own interpreter |
| runaway/hung code | wall-clock timeout kills the whole process group; memory, CPU-time and file-size limits; output read through a bounded buffer |
| model cheats by editing tests | tests are locked while implementing and rewritten from the approved copy before commit |
| half-finished work | per-step snapshot/rollback; atomic file writes; crash-safe progress file |
| RAM / heat / battery / disk | resource guard pauses before every model call (min free RAM, battery % when unplugged, CPU temp, free disk) and resumes by itself |
| secrets leaking into generated code's environment | `*TOKEN*`, `*SECRET*`, `*PASSWORD*`, `*API_KEY*` env vars are not passed to commands |
| code leaving the machine | refuses any non-loopback model URL unless `allow_remote_llm = true` |
| two agents in one folder | exclusive lock per workspace |
| git surprises | commits only the step's files; never prompts, never signs (no hang), never runs your hooks |

## Configuration (`codeagent.toml`, all keys optional)

Defaults are tuned for a 2 GB GPU / 12 GB RAM laptop. Most useful keys:

| key | default | meaning |
|---|---|---|
| `model` | `qwen3.5:0.8b` | name from `ollama list` (env: `CODEAGENT_MODEL`) |
| `base_url` | `http://localhost:11434` | model server (env: `CODEAGENT_BASE_URL`) |
| `backend` | `ollama` | or `openai` for llama.cpp's server |
| `num_ctx` / `num_predict` | `4096` / `1536` | context / max reply tokens (smaller = less VRAM) |
| `temperature` | `0.2` | try `0` if `smoke` is flaky |
| `max_attempts` | `4` | retries per phase before a step is rolled back |
| `stop_on_failure` | `true` | stop the plan at the first failed step |
| `step_cooldown_s` | `2.0` | pause after every model call (keeps the laptop cool) |
| `min_free_mem_mb` / `min_battery_pct` / `max_temp_c` / `min_free_disk_mb` | `1500` / `20` / `88` / `1024` | when to pause |

## Honest expectations

* The whole system is tested (260+ tests, including real subprocess/HTTP/git end-to-end runs),
  but against a **scripted stand-in for the model**. How well *your* 0.8B model does is
  something only `codeagent smoke` and real plans can tell you.
* A 0.8B model will fail some steps. That is expected and safe: the step is rolled back,
  the reason is logged, and the plan can be re-run. Smaller steps and clearer examples help most.
* If it is too weak, point `model` at a bigger local model; nothing else changes.

## Layout

`plan.py` plan format · `runner.py` the loop · `prompts.py` · `parsing.py` model output ·
`llm.py` Ollama/OpenAI-compatible client · `shell.py` hardened command runner ·
`workspace.py` sandboxed files · `git.py` · `resources.py` guard · `state.py` resume + lock ·
`daemon.py` 24/7 inbox · `cli.py`
