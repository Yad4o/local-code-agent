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
export CODEAGENT_MODEL="<name from ollama list>"

codeagent doctor                    # checks Ollama, the model, git, RAM/battery/temp/disk
codeagent smoke --tries 5           # can YOUR model follow the output format? (needs >= 3/5)
codeagent template --out plan.json  # a plan skeleton (see tests/examples/)
codeagent check plan.json           # validate a plan without running anything
codeagent run plan.json --workspace ~/agent-projects/demo
```

Everything lives in one module, `agent.py` (`python -m agent` works too). Exit codes of `run`: `0` done,
`1` a step failed, `2` bad plan/settings, `3` paused (model server down or system resources low; just run it
again, finished steps are skipped), `4` workspace busy.

## Thinking

Agentic coding needs the model to reason before it writes code, so `think` is **on** by default
(`think_level = high` for models that take a level, such as gpt-oss; other models take on/off).
Thinking tokens count against `num_predict`, so the defaults are larger (4096 / 2048) than for a
non-thinking model. If a reply ends with "used all N tokens thinking", raise `CODEAGENT_NUM_PREDICT`
or lower `CODEAGENT_THINK_LEVEL`. Switch it off with `CODEAGENT_THINK=false`.

## Plans

A plan is a small JSON file with one fixed structure, validated strictly before anything runs (all
mistakes are reported at once). Every step names **exactly** the files it may touch:

```json
{
  "title": "Slugify utility",
  "test_command": "python -m pytest -q",
  "steps": [{
    "title": "slugify lowercases and joins words with dashes",
    "description": "Create slugify(text) in slugify/core.py ... 'Hello, World!' -> 'hello-world'.",
    "commit": "feat: add slugify core function",
    "tests": ["tests/test_core.py"],
    "files": ["slugify/__init__.py", "slugify/core.py"]
  }]
}
```

**Let the plan decide what is correct (recommended for small models).** A tiny model that also writes the
tests invents contradictory ones. Give the step a `function` and `examples` and the agent prints the test
file itself, so the model only writes the implementation (see `tests/examples/cart.json`):

```json
"function": "cart.engine:add_item",
"examples": [
  {"args": [[], "apple", 1.5], "expect": [{"name": "apple", "price": 1.5, "quantity": 1}]},
  {"args": [[], "apple", -1.0], "raises": "ValueError"}
]
```

Values must be plain JSON (strings, numbers, booleans, lists, objects). Floats are compared exactly. Add
`"signature": "add_item(cart, name, price, quantity=1)"` and the harness writes the `def` line too, so the
model writes only the body. Tips for a tiny model: **one function per step**, exact input -> output
examples. The agent works on a branch `agent/<plan-title>`, never on your `main`.

## Watching it work / finding out why a step failed

**Live, in the same terminal:** add `--live` to `run` to see each prompt that goes to the model and its
raw output (thinking first, then the answer) as it streams. It is shown in the terminal only and is not
saved anywhere. Ollama backend only (the `openai` backend prints the whole reply when it arrives).

```bash
codeagent run plan.json --workspace ~/agent-projects/demo --live
```

Without `--live`, the log still prints `asking the model (...)` before every call, so a long
reasoning pause is not mistaken for a hang.

**Everything in one file:** every run (with or without `--live`) appends each prompt and the model's
raw reply to `<workspace>/.codeagent/transcript.md`, in order, with seconds / cut-off / temperature.
The model's thinking is included too when you run with `--live` (that is when it is streamed).

```bash
less $W/.codeagent/transcript.md
```

### When the model loops

Small thinking models sometimes go round in circles (a phrase again and again, or a whole paragraph).
Every call is now streamed, and a loop guard watches it: if the model repeats itself, or thinks for more than
`CODEAGENT_MAX_THINK_CHARS` (default 8000) without starting an answer, the call is stopped at once instead of
burning all `num_predict` tokens. That counts as a failed attempt, not a pause: the next attempt gets a warmer
temperature, a note saying what went wrong, and **thinking switched off** for that call. Switch the guard off
with `CODEAGENT_LOOP_GUARD=false`. Whatever streamed in before a stop is still saved in `transcript.md`.

Writing a function body does **not** think by default (`CODEAGENT_THINK_BODY=false`, with
`CODEAGENT_TEMPERATURE_NO_THINK=0.3`): in a real run a 0.8B model spent 4-5 minutes wandering in its thinking
for a 5-line body, while the same call without thinking answered in 8 seconds. Set `CODEAGENT_THINK_BODY=true`
to bring thinking back. A body that still fails without thinking is retried with thinking from attempt
`CODEAGENT_THINK_BODY_AFTER` (default 0 = never; try 4), still bounded by `CODEAGENT_MAX_THINK_CHARS`. The harness also writes the `from pkg.mod import name` lines for every context function
named in the step description, and always puts the planned signature back if the model rewrote the def line.

Files the step description names (like "use check_args from cart/checks.py") are shown to the model and imported
for it even if the plan's `context` forgot them. A retry that hides the model's old code still shows those files,
and the failing-test report names the real call (`add_item([], 'a') must raise ValueError, but it returned normally`)
instead of `test_example_4`. `CODEAGENT_MAX_ATTEMPTS` now defaults to 6, because calls without thinking take seconds.

Settings that matter for loops: `CODEAGENT_TEMPERATURE` (about 0.6 for thinking models; too low loops more),
`CODEAGENT_TOP_P` / `CODEAGENT_TOP_K` (0.95 / 20 by default), `CODEAGENT_REPEAT_PENALTY` (1.1),
`CODEAGENT_NUM_PREDICT`. `CODEAGENT_THINK_LEVEL` only affects gpt-oss models; other models think on or off.


```bash
W=~/agent-projects/demo                                   # your --workspace
tail -f $W/.codeagent/agent.log                           # live log (also printed in the terminal)
ls $W/.codeagent/replies/                                 # EVERY raw model reply, newest last
cat $W/.codeagent/replies/*-tests-a1.txt                  # header says: seconds, cut off yes/no, temperature
```

`model replied in 197s ... CUT OFF at the token limit` means the model rambled until it hit `num_predict`.
The agent then keeps the longest valid-Python prefix (or asks for a SHORTER file), and every retry runs
slightly warmer (capped) so a small model is nudged out of a loop without turning to nonsense.

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
| code leaving the machine | refuses any non-loopback model URL unless `CODEAGENT_ALLOW_REMOTE_LLM=true` |
| two agents in one folder | exclusive lock per workspace |
| git surprises | commits only the step's files; never prompts, never signs (no hang), never runs your hooks |

## Settings

Defaults are tuned for a 2 GB GPU / 12 GB RAM laptop. There is no config file yet: every setting of the
`Config` dataclass in `agent.py` can be overridden with an environment variable `CODEAGENT_<NAME>`
(lists are comma separated). The useful ones:

| variable | default | meaning |
|---|---|---|
| `CODEAGENT_MODEL` | `qwen3.5:0.8b` | name from `ollama list` |
| `CODEAGENT_BASE_URL` / `CODEAGENT_BACKEND` | `http://localhost:11434` / `ollama` | model server; backend `openai` for llama.cpp's server |
| `CODEAGENT_THINK` / `CODEAGENT_THINK_LEVEL` | `true` / `high` | reasoning before code (see Thinking) |
| `CODEAGENT_NUM_CTX` / `CODEAGENT_NUM_PREDICT` | `8192` / `4096` | context / max tokens per call, thinking included |
| `CODEAGENT_TEMPERATURE` | `0.2` | retries warm up by `RETRY_TEMPERATURE_STEP` (0.05) up to `MAX_RETRY_TEMPERATURE` (0.4) |
| `CODEAGENT_MAX_ATTEMPTS` | `4` | retries per phase before a step is rolled back |
| `CODEAGENT_MAX_TEST_REWRITES` | `2` | times tests found broken while implementing are rewritten |
| `CODEAGENT_STOP_ON_FAILURE` | `true` | stop the plan at the first failed step |
| `CODEAGENT_STEP_COOLDOWN_S` | `2.0` | pause after every model call (keeps the laptop cool) |
| `CODEAGENT_MIN_FREE_MEM_MB` / `_MIN_BATTERY_PCT` / `_MAX_TEMP_C` / `_MIN_FREE_DISK_MB` | `1500` / `20` / `88` / `1024` | when to pause |

## Honest expectations

* The system is tested with real subprocess/HTTP/git end-to-end runs, but against a **scripted stand-in
  for the model**. How well *your* model does is something only `codeagent smoke`, `tests/bench.py` and
  real plans can tell you.
* A 0.8B model will fail some steps. That is expected and safe: the step is rolled back, the reason is
  logged, and the plan can be re-run. Smaller steps and clearer examples help most; a bigger local model
  needs only a different `CODEAGENT_MODEL`.

## Layout

```
agent.py          config, plan, prompts, log, the agent loop and the CLI: one module on purpose
tests/            unit + end-to-end tests, example plans (tests/examples/*.json), bench.py
```

Background services (a watch-folder daemon, systemd unit) are deliberately not here yet: first the agent has
to do a task well on its own.

## Tools around the model

The model is a fast typist that guesses; code does the checking and the tidying.

| Tool | What it does | Switch |
|---|---|---|
| Examples as tests | the plan's `examples` become the pytest file; the model never writes tests | `function` + `examples` in the plan |
| Skeleton mode | the plan's `signature` gives the def line; the harness writes it and `__init__.py`, the model writes only the body (cap `num_predict_body`) | `signature` in the plan |
| Tidy | strips docstrings and comments and re-prints the code in one style (stdlib `ast`) | `CODEAGENT_TIDY_CODE` (default on) |
| ruff (optional, `pip install ruff`) | removes unused imports, reports undefined names / redefinitions / syntax errors with line numbers before pytest runs | `CODEAGENT_USE_RUFF` (default on, skipped when ruff is missing) |
| Function-defined check | the implementation must really define the function the examples call | automatic |
| Duplicate / parameter checks | model-written tests with repeated names or parameters nothing provides are rejected | automatic |
| Plain-language failures | each failing example says `add_item(...) returned X, but it must return Y` instead of a raw pytest diff the model cannot read | automatic |
| Header-less reply | a single fenced block with no `FILE:` line is used when the step can only have one file | automatic |
| Never-returns check | an implementation with no `return <value>` is sent back when the examples expect a result | automatic |
| Repeated-code check | identical code to an earlier failed attempt is not re-run; the next prompt hides the old code and runs a little warmer | automatic |
| Repeat penalty + stop sequence | `repeat_penalty` (Ollama) fights "same sentence forever" loops; skeleton mode stops at a stray `FILE:` | `CODEAGENT_REPEAT_PENALTY` |

See `tests/examples/cart.json` for a step using all of it.

### Small steps beat one big step

A 0.8B model can write a 5-line function from a numbered recipe; it cannot invent the algorithm for a
compound one. `tests/examples/cart_split.json` splits the cart into three tiny steps (validate, merge, add),
each with its algorithm written in plain words. Later steps see earlier files through `context`, and each
skeleton step adds its re-export to the package `__init__.py` instead of replacing it.

### Writing step descriptions a small model can follow

The model cannot invent an algorithm; it can follow a recipe. Write the recipe: numbered lines, every variable
named, real expressions. A description that said "append a copy of the item whose quantity is old + new" failed
all 6 attempts (the model appended the same item in both branches); the same step written as
`new_item = dict(item); new_item["quantity"] = item["quantity"] + quantity; result.append(new_item)` passed on
attempt 1. Put error checks first, state tie-breaking rules, avoid words like "validate" or "handle edge cases",
give every loop its own line (never `for ...: if ...:` on one line), keep a recipe to 5 lines and one loop (split
it otherwise), and give an example for every branch and every `raises`. Examples cannot expect `null`/`None`, tuples or sets.
The full rules are in [`docs/PLAN_GUIDE_FOR_AI.md`](docs/PLAN_GUIDE_FOR_AI.md): give that file to any
AI assistant, then tell it what you want built, and it writes the plan JSON for you.

More example plans: `tests/examples/cart_totals.json` (steps that call earlier steps), `text_tools.json`
(strings, loops, a hard step split in two), `gradebook.json` (branch chains, dict input, errors). Run any of them with
`codeagent run tests/examples/<name>.json --workspace ~/agent-projects/<name> --live`.
`codeagent check plan.json` also prints `warning:` lines for description lines a small model copies wrongly (two
assignments joined with `and`, a loop and its body on one line, `;`-joined statements, English conditions, recipes
longer than 6 lines or with two loops); fix them before running.

### Building something big

Split it into **pure functions** (data in, data out) and build those with plans; write the glue that touches the
outside world (files, network, `print`, CLI, classes) yourself. Keep one plan to 5-15 steps; for a bigger project
make several plans (core, then rules, then formatting...) and run them in order **in the same workspace**: each
new plan branches from the previous one, so its steps can use the earlier files through `context`. Finished steps
are skipped on a rerun, and a step you edit runs again.

## Is the model the problem? `tests/bench.py`

Tests the model with no agent around it: seven small functions asked in plain words (3 basic, 3 medium,
1 hard), each sampled several times and graded by running the code.

```bash
python tests/bench.py                          # 5 tries per problem
python tests/bench.py --tries 3 --only add slugify add_item
CODEAGENT_MODEL="qwen2.5-coder:1.5b" python tests/bench.py     # compare another model
```

It ends with a plain verdict: too weak for code, fine on tiny functions only, needs small steps, or capable.
