# How to write a build plan for the local code agent

**Instructions for an AI assistant.** The user will give you this file and then tell you what they want built
(an app, a library, a tool). Your job is to turn that request into one or more **plan JSON files** for their local
coding agent. A tiny local model (about 0.8B parameters) will write the code, one small function at a time, so the
plan has to be written for a model that is weak at inventing algorithms and good at following a recipe.

Read this whole file before you write anything. The rules come from real failed runs.

---

## 1. What you deliver

Reply with exactly these parts, in this order, and nothing else:

1. **Architecture (5-15 lines).** The package name, the data shapes, and the list of functions with a
   one-line purpose each, in build order.
2. **The plan JSON**, in one ```json code block. If the project needs more than 15 steps, deliver it as several
   plans (see section 9), each in its own code block with a file name like `plan-1-core.json`.
3. **`project.json`**: the one file the user builds from (format in section 9a).
4. **`groups.json`**: how the one-module-per-function files are merged into a few themed modules afterwards
   (format in section 9a).
5. **Hand-written parts.** Everything that cannot be a step (section 4): one line per file, saying what it does and
   which agent-built functions it calls. Do not write that code unless the user asks for it.
6. **Commands to run**, copied from section 10 with the real file names filled in.

If the request is vague, make sensible choices and state them in the architecture part. Ask a question only when a
wrong guess would waste the whole plan.

---

## 2. How the agent works (why the rules below exist)

* The agent builds **one step at a time**. A step creates **one top-level function**.
* The agent itself writes the test file from your `examples`. The model never writes tests and never decides what is
  correct. **Your examples are the only specification.** A wrong example makes a step impossible to pass.
* The model sees only: the step `description`, the function line, the `examples`, the files in `context`, and what
  the last attempt did wrong. It does **not** see the rest of the plan or your architecture. Every step has to be
  understandable on its own.
* The harness writes the `def` line (from `signature`) and the imports of earlier functions. The model writes only
  the body.
* After a failed attempt the harness compares the recipe's **Python lines** with the model's body and tells it which
  of them are missing ("copy them as they are"). So every recipe line should be exact, valid Python.
* The model gets 6 attempts. If it fails, the agent tries **your recipe lines as code** (when every line is valid
  Python the function is built from the recipe itself), then a different "repair this draft" prompt. Only then is the
  step parked: the run goes on with the steps that do not need it and the failure is written to a report (section 12).
  So the safest recipe is one whose every line is valid Python.
* With `codeagent build` the agent works on one branch `agent/<project>`, commits every step as the user, pushes to a
  private GitHub repo, and fast-forwards `main` only when everything is built and merged. It never commits secrets.

---

## 3. Plan JSON format

```json
{
  "title": "Cart Totals",
  "test_command": "python -m pytest -q",
  "steps": [
    {
      "title": "line_total multiplies price by quantity",
      "description": "Create line_total(item) in totals/line.py. item is a dict {\"name\", \"price\", \"quantity\"}.\n1. return item[\"price\"] * item[\"quantity\"]\n",
      "commit": "feat: add line_total",
      "tests": ["tests/test_line.py"],
      "files": ["totals/__init__.py", "totals/line.py"],
      "context": [],
      "function": "totals.line:line_total",
      "signature": "line_total(item)",
      "examples": [
        {"args": [{"name": "apple", "price": 1.5, "quantity": 2}], "expect": 3.0},
        {"args": [{"name": "kiwi", "price": 0.5, "quantity": 0}], "expect": 0.0}
      ]
    }
  ]
}
```

### Rules the checker enforces (a plan that breaks one is rejected)

* Top-level keys: `title`, `test_command`, `steps` (and optional `workspace`). Nothing else.
* `test_command` is always `"python -m pytest -q"`.
* Step keys: `title`, `description`, `commit`, `tests`, `files`, `function`, `signature`, `examples`, and optional
  `context`. Nothing else. Always give all of `function`, `signature` and `examples`.
* `title`: unique in the plan. `commit`: one line, at most 72 characters, like `feat: add line_total`.
* `function`: `package.module:function_name`, for example `totals.line:line_total`.
* `files`: **exactly** two paths: the module file and the package `__init__.py`
  (`["totals/__init__.py", "totals/line.py"]`). Never anything else.
* `tests`: **exactly one** path, unique per step: `tests/test_<module>.py`.
* `context`: files from **earlier steps** that this step's function uses. Use `[]` when there are none.
* `signature`: the signature without `def` and without a colon, with the same parameter names as the description:
  `"add_item(cart, name, price, quantity=1)"`.
* Paths use `/`, are relative, and never contain `..`.
* Each example has `args` (a list), optional `kwargs` (an object), and **exactly one** of `expect` (the result) or
  `raises` (a builtin exception name such as `"ValueError"`, `"KeyError"`, `"TypeError"`).
* Example values are plain JSON: strings, numbers, booleans, lists, objects.
  * **No `null`.** A function whose correct result is `None` cannot be an example. Make it raise an error or return
    something else instead.
  * **No tuples, sets, or non-string dict keys.** Return lists, sorted lists, and string keys.
  * **Floats are compared exactly.** Use numbers that are exact in binary (`1.5`, `0.25`, `200.0`) or use
    integers. Never use sums like `0.1 + 0.2`.
* Keep each example under about 300 characters once printed (the prompt cuts longer ones).

Run `python -m agent check plan.json` (or `codeagent check plan.json`) to validate the file. Tell the user to do it.

---

## 4. What can be a step, and what cannot

A step must be a **pure function**: data in, data out, same answer every time.

**Can be a step:** parsing a string, formatting text, math, filtering/sorting/merging lists and dicts, validation
that raises errors, a small state change that returns a **new** value, a decision (`if`/`elif`) chain, anything you
can describe with examples of the form "these arguments give this result".

**Cannot be a step** (put them in "Hand-written parts"; keep them thin, they just call the tested functions):

* classes and methods (only top-level functions)
* reading or writing files, databases, network, environment variables
* `print`, `input`, command-line parsing, GUI/web handlers, `main()`
* time, dates "now", randomness, threads, async
* functions that change their arguments in place and return nothing
* anything that needs mocks

Design the pure core first. A todo app, for example, becomes `add_task(tasks, text)`, `complete_task(tasks, id)`,
`filter_tasks(tasks, status)`, `format_task(task)` (all pure, built by the agent) plus a hand-written `store.py` and
`cli.py` that load/save JSON and print. Represent state as plain lists and dicts and **return new copies**.

---

## 5. How to break the work into steps

1. **Fix the data shapes first** and write them out literally: `{"id": 1, "text": "milk", "done": false}`. Copy the
   same literal shape into every description that uses it (each step sees only its own description).
2. **List the functions bottom-up**: small helpers first, functions that call them later.
3. **One function per step.** Its body should be at most about 8 lines, with at most one loop. If you cannot write
   the recipe in 6 short numbered lines, split the function into two steps.
4. **A function may call at most 2-3 earlier functions.** Every function it calls must come from an earlier step and
   be listed in `context`.
5. **Order matters**: a step may only use files created by earlier steps. If a step names a file that does not exist
   yet, it stops immediately with a "missing file" message.
6. **Layout**: one package for the plan, one module per function (`totals/line.py`, `totals/cart.py`, ...), module
   names unique within the plan, test files `tests/test_<module>.py`. Every step lists the shared
   `<package>/__init__.py`; the harness adds each function's re-export to it.

---

## 6. How to write the `description` (the most important part)

The model cannot think its way to an algorithm. **Write the algorithm for it, as a recipe of tiny statements.**

Template:

```
Create <exact signature> in <module path>. <Data shape, with a literal example if it is a dict or list.>
<Return a NEW list/dict; never change the input.>   (only when it applies)
<Tie-break or edge rules in one sentence.>          (only when they apply)
1. <error checks first: "If ..., raise ValueError.">
2. <variable = starting value>
3. For each x in y:
   - if <condition>: <exact statements>
   - else: <exact statements>
4. return <expression>
```

Rules:

* Start with `Create name(params) in path/to/module.py.`, with the same signature as the `signature` field.
* **Name every variable** and write real expressions. Write `new_item = dict(item)` and
  `new_item["quantity"] = item["quantity"] + quantity`. Never write "make a modified copy".
* **One statement per line.** Write a loop as its own line (`For each ch in text:`) with the statements under it as
  `-` bullets. Never put a `for` and an `if` on the same line, and never join two statements with `and` or `;`
  (the model copies the shape literally and writes invalid Python such as
  `for ch in text.lower(): if ch.isalnum(): ...`). A recipe line `best_name = None and best_avg = None` was copied
  verbatim into the code and failed all 6 attempts: write `best_name = None` and `best_avg = None` as two lines. Branches are `- if ...:` and `- else:` bullets; an `if` that
  needs two statements gets one nested `-` bullet per statement.
* Put error checks first, in the order the examples test them.
* **Never reassign a parameter.** Give the cleaned value a new name. `1. category = category.strip().lower()` then
  `2. if category == "": raise ValueError` failed all 6 attempts: the model folded step 1 into the return and checked
  the untrimmed text. `1. cleaned = category.strip().lower()`, `2. if cleaned == "": raise ValueError`,
  `3. return cleaned` leaves nothing to fold.
* **A standard-library import is its own first recipe line**: `1. import re` (several: `1. import re, decimal`), so the
  model does not forget it. The harness moves it to the top of the module.
* For tie-breaking ("alphabetically first", "smallest ID") a short loop over `sorted(...)` with a strict `>` is easier
  for a small model than a `sorted(..., key=lambda x: (-x["amount"], x["id"]))` with a negated key.
* **Assign the result of a helper call even when you only want its error**: `found = find_transaction(transactions,
  transaction_id)`, not a bare `find_transaction(...)` line. A line whose value is never used looks optional to a
  small model, and it drops the line (it dropped `raise KeyError` checks twice).
* **Write checks and early returns as code, not English.** `2. if len(text) <= limit: return text` is followed;
  `2. If the text already fits, return it.` was skipped on all 6 attempts (the model kept the other two lines and
  dropped that one). Use real conditions (`if len(scores) == 0: raise ValueError`), one short statement after the colon.
* **Keep a recipe short: at most 6 numbered lines and one loop** (`codeagent check` warns beyond that). A 0.8B model skips steps in a longer recipe
  (a 6-line, two-loop `most_common_word` failed all 6 attempts: it replaced `best = order[0]` with
  `best = None` and dropped an update). Split it instead: `word_counts(text)` in one step, then
  `most_common_word(text)` using `word_counts` in the next. Thinking mode did not rescue it.
* State tie-breaking explicitly ("if two words are used equally often, the one that appears first wins").
* To use an earlier function write `Use line_total from totals/line.py (already written, shown below).` and put
  that file in `context`. Only name files that earlier steps create.
* To use the standard library name the module (`use re`); the model imports it. Do not tell it to import your own
  earlier functions, the harness does.
* Do **not** use vague words: "validate", "handle edge cases", "robust", "appropriately", "etc.", "efficient".
  Do **not** mention tests, and do not ask for docstrings or comments.
* Never ask for `print`, tuples, `None` as a result, or in-place changes.

### Real before and after

This step failed all 6 attempts. The model wrote `append(item)` in both branches and never made the changed copy:

```
2. For each item in cart: if item["name"] equals name, append a copy of the item whose quantity is
   item["quantity"] + quantity; otherwise append the item unchanged.
```

The same step, rewritten as a recipe, passed on attempt 1 (the one-line form below worked; one statement per line
is safer still):

```
1. result = []
2. For each item in cart:
   - if item["name"] == name:
     - new_item = dict(item)
     - new_item["quantity"] = item["quantity"] + quantity
     - result.append(new_item)
   - else:
     - result.append(item)
3. return result
```

---

## 7. How to write the `examples`

The examples are the tests. Use 3-6 per step:

* one example per branch of the recipe (every `if`/`elif`/`else`);
* **every `raises` rule in the recipe** gets its own example;
* boundary values (`0`, empty list, the exact limit such as `90` for "at least 90");
* at least one example where doing nothing (returning the input unchanged) would be **wrong**;
* small inputs, so they are easy to check by hand.

Compute every expected value by hand, twice. The agent trusts your examples completely. For a calculation, redo it
step by step and check the result is exact in floating point.

---

## 8. Complete small example (3 steps)

```json
{
  "title": "Shopping Cart Engine (small steps)",
  "test_command": "python -m pytest -q",
  "steps": [
    {
      "title": "check_args validates price and quantity",
      "description": "Create check_args(price, quantity) in cart/checks.py.\n1. if price < 0: raise ValueError\n2. if quantity <= 0: raise ValueError\n3. return True\n",
      "commit": "feat: add cart check_args",
      "tests": [
        "tests/test_checks.py"
      ],
      "files": [
        "cart/__init__.py",
        "cart/checks.py"
      ],
      "function": "cart.checks:check_args",
      "signature": "check_args(price, quantity)",
      "examples": [
        {
          "args": [
            1.5,
            1
          ],
          "expect": true
        },
        {
          "args": [
            0.0,
            1
          ],
          "expect": true
        },
        {
          "args": [
            -1.0,
            1
          ],
          "raises": "ValueError"
        },
        {
          "args": [
            1.5,
            0
          ],
          "raises": "ValueError"
        }
      ]
    },
    {
      "title": "merge_quantity raises the quantity of one item",
      "description": "Create merge_quantity(cart, name, quantity) in cart/merge.py. cart is a list of dicts {\"name\", \"price\", \"quantity\"}.\nReturn a NEW list; never change the input cart.\n1. result = []\n2. For each item in cart:\n   - if item[\"name\"] == name:\n     - new_item = dict(item)\n     - new_item[\"quantity\"] = item[\"quantity\"] + quantity\n     - result.append(new_item)\n   - else:\n     - result.append(item)\n3. return result\n",
      "commit": "feat: add cart merge_quantity",
      "tests": [
        "tests/test_merge.py"
      ],
      "files": [
        "cart/__init__.py",
        "cart/merge.py"
      ],
      "function": "cart.merge:merge_quantity",
      "signature": "merge_quantity(cart, name, quantity)",
      "examples": [
        {
          "args": [
            [
              {
                "name": "apple",
                "price": 1.5,
                "quantity": 1
              }
            ],
            "apple",
            2
          ],
          "expect": [
            {
              "name": "apple",
              "price": 1.5,
              "quantity": 3
            }
          ]
        },
        {
          "args": [
            [
              {
                "name": "apple",
                "price": 1.5,
                "quantity": 1
              }
            ],
            "kiwi",
            1
          ],
          "expect": [
            {
              "name": "apple",
              "price": 1.5,
              "quantity": 1
            }
          ]
        }
      ]
    },
    {
      "title": "add_item adds an item to the cart",
      "description": "Create add_item(cart, name, price, quantity=1) in cart/engine.py. Use check_args from cart/checks.py and\nmerge_quantity from cart/merge.py (already written, shown below). Return a NEW list; never change the input cart.\n1. check_args(price, quantity)\n2. if any(item[\"name\"] == name for item in cart): return merge_quantity(cart, name, quantity)\n3. return [*cart, {\"name\": name, \"price\": price, \"quantity\": quantity}]\n",
      "commit": "feat: add cart add_item",
      "tests": [
        "tests/test_engine.py"
      ],
      "files": [
        "cart/__init__.py",
        "cart/engine.py"
      ],
      "context": [
        "cart/checks.py",
        "cart/merge.py"
      ],
      "function": "cart.engine:add_item",
      "signature": "add_item(cart, name, price, quantity=1)",
      "examples": [
        {
          "args": [
            [],
            "apple",
            1.5
          ],
          "expect": [
            {
              "name": "apple",
              "price": 1.5,
              "quantity": 1
            }
          ]
        },
        {
          "args": [
            [],
            "apple",
            -1.0
          ],
          "raises": "ValueError"
        }
      ]
    }
  ]
}
```

More complete plans to copy the style from: `tests/examples/cart_totals.json` (steps that use earlier
steps), `tests/examples/text_tools.json` (strings, loops, a split of a hard step into `word_counts` + `most_common_word`), `tests/examples/gradebook.json`
(branch chains, dict input, errors).

---

## 9. Plan size

* One plan: **5-15 steps** is comfortable, 30 is the upper limit. A failure stops the run, and the user fixes it
  and reruns; finished steps are skipped.
* For a bigger project, deliver **several plans**, one per layer or module (core data functions, then rules, then
  formatting...). Give each plan a **different title** and a different package, or the same package with different
  module names. Plan 2 may list files built by plan 1 in `context`, because all plans run in the same workspace,
  in order.
* Never put two unrelated features in one step to save space. More, smaller steps are always better.

---

## 9a. project.json and groups.json

`project.json` (the file the user builds from; paths are relative to it):

```json
{
  "name": "expense-manager",
  "description": "One sentence about what the project is.",
  "plans": ["plan-1-core.json", "plan-2-reports.json"],
  "groups": "groups.json"
}
```

* `name` becomes the private GitHub repository: lower case, words joined with dashes, short and descriptive.
* `plans` lists the plan files in build order. Only these keys are allowed (and an optional `"workspace"`; leave it out).

`groups.json` merges the many small modules into a few themed ones after the build:

```json
{
  "package": "expense_manager",
  "modules": {
    "validation": ["amounts", "categories", "dates"],
    "records": ["make", "ids", "find"],
    "reports": ["sum_amount", "monthly_summary"]
  }
}
```

* `package` is the package name used in every step's `function`. The lists hold the **module** names (the part
  between the dots in `function`: `expense_manager.amounts:validate_amount` is module `amounts`).
* Every module of every plan goes into exactly one group. Aim for 3-8 groups with themed names.
* **Imports must only go one way.** Order the groups from foundation to top; a module may only call functions from
  its own group or from groups listed **before** it. If two groups would need each other, merge them.

---

## 10. Commands the user will run

```bash
codeagent setup                          # once: saves the GitHub token in ~/.codeagent.env (asks for it, hidden)
codeagent check plan-1-core.json         # for each plan; fix every warning
codeagent build project.json             # builds every plan, merges the modules, publishes to a private repo
```

`codeagent build` does everything: it creates the private repository, builds the plans in order, rescues or parks
failing steps instead of stopping, merges the modules with `groups.json`, then pushes the branch and `main`. A
rerun continues where it stopped. `codeagent build project.json --no-github` builds locally only.

`codeagent check` also prints `warning:` lines for description lines a small model tends to copy wrongly (joined
statements, a loop and its body on one line, English conditions, recipes that are too long): fix every warning
before building. A step whose description or examples were edited runs again.

If steps fail, `build` writes `FAILURES.md` and prints what to do: the user pastes that file into an AI chat, saves
the answer as `fix.json`, runs `codeagent fix fix.json --workspace <folder>` and builds again. Section 12 says how
to answer such a report.

---

## 11. Self-check before you reply

Go through every step and answer yes to each:

- [ ] One top-level function; no class, I/O, `print`, randomness or time.
- [ ] `signature` matches the first line of the description, and `function` is `package.module:name`.
- [ ] `files` is exactly `[package/__init__.py, package/module.py]`; `tests` is one unique path.
- [ ] Every function used by the step comes from an **earlier** step and is in `context`.
- [ ] The description is a numbered recipe with named variables and real expressions; no vague words.
- [ ] Data shapes are written out literally in the description.
- [ ] Every branch and every `raises` has an example; each expected value was checked by hand.
- [ ] No `null`, tuple, set or inexact float anywhere in the examples.
- [ ] The body is at most about 8 lines, with at most one loop; the recipe has at most 6 numbered lines.
- [ ] No recipe line reassigns a parameter, and every helper call whose value is unused is assigned to a variable.
- [ ] Every loop is on its own line with its statements as bullets underneath; no `and`/`;` joining statements.
- [ ] Titles and commits are unique/valid; commit lines are at most 72 characters.
- [ ] `project.json` lists every plan in build order; `groups.json` places every module in exactly one group, and
      no group needs a function from a group listed after it.
- [ ] Everything that is not a pure function is listed under "Hand-written parts".

---

## 12. When the user pastes a failure report (`FAILURES.md`) or a failing transcript

Answer with **one JSON object, a fix pack, and nothing else**. The user saves it and runs `codeagent fix fix.json`.

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
(the lesson then applies to every project). Reply with the JSON object and nothing else.

How to choose what to put in it:

* **Almost always rewrite `plan_step.description`** using the rules in sections 5-6 (and an `examples` list only when
  an example was wrong). Look at what the model wrote in its last attempts: it usually dropped, merged or rewrote a
  recipe line. Write that line so it cannot be misread (as code, on its own line, with a new variable name).
* Keep the step's title; never rename it and never touch `function`, `files` or `tests`.
* Use `body` only when you are certain of the exact function body; it is tried first on the next run and must still
  pass the tests.
* Add a `lesson` when the mistake could repeat in other steps ("Never reassign a parameter; use a new variable name.").
  One short rule each, at most 5 per fix. Use `"scope": "global"` for rules that hold for every project.
* Fix every failing step of the report in the same pack (one entry per failure id).
