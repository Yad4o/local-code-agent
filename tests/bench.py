"""Raw model test: can this model write these small functions AT ALL, with no agent around it?

Each problem is asked in plain words, like a person typing into a chat. The reply's code is run
against hidden examples. Each problem is sampled several times, so you see "3/5" rather than luck.
Run it with `python tests/bench.py`, then read the verdict at the end.
"""
import argparse
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import (
    Example,
    LLMError,
    Step,
    load_config,
    make_client,
    render_tests,
    run_command,
)

CART = [{"name": "apple", "price": 1.5, "quantity": 1}]


@dataclass(frozen=True)
class Problem:
    key: str
    level: str                     # "basic" | "medium" | "hard"
    function: str
    task: str                      # what the person types; no harness tricks
    examples: tuple[Example, ...]


def _ex(*args, expect=None, raises=None, **kwargs) -> Example:
    return Example(tuple(args), kwargs or None, expect, raises)


PROBLEMS: tuple[Problem, ...] = (
    Problem("add", "basic", "add",
            "Write a Python function add(a, b) that returns the sum of a and b.",
            (_ex(1, 2, expect=3), _ex(-1, 1, expect=0), _ex(0.5, 0.25, expect=0.75))),
    Problem("is_even", "basic", "is_even",
            "Write a Python function is_even(n) that returns True if the integer n is even, otherwise False.",
            (_ex(4, expect=True), _ex(7, expect=False), _ex(0, expect=True))),
    Problem("slugify", "basic", "slugify",
            "Write a Python function slugify(text) that lowercases text and replaces every run of "
            "non-alphanumeric characters with a single hyphen, with no hyphen at the start or end.",
            (_ex("Hello, World!", expect="hello-world"), _ex("  A  B ", expect="a-b"), _ex("", expect=""))),
    Problem("fizzbuzz", "medium", "fizzbuzz",
            "Write a Python function fizzbuzz(n) that returns a list of strings for 1..n: 'Fizz' for "
            "multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for both, otherwise the number as a string.",
            (_ex(5, expect=["1", "2", "Fizz", "4", "Buzz"]),
             _ex(15, expect=["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz", "11", "Fizz",
                             "13", "14", "FizzBuzz"]))),
    Problem("count_words", "medium", "count_words",
            "Write a Python function count_words(text) that returns a dict mapping each lowercase word "
            "(split on whitespace) to how many times it appears.",
            (_ex("a b a", expect={"a": 2, "b": 1}), _ex("Hi hi HI", expect={"hi": 3}), _ex("", expect={}))),
    Problem("merge_quantity", "medium", "merge_quantity",
            "Write a Python function merge_quantity(cart, name, quantity). cart is a list of dicts with keys "
            "name, price, quantity. Return a NEW list in which the item called name has its quantity increased "
            "by quantity; other items stay the same. Do not modify the input list.",
            (_ex(CART, "apple", 2, expect=[{"name": "apple", "price": 1.5, "quantity": 3}]),
             _ex(CART + [{"name": "pear", "price": 2.0, "quantity": 2}], "pear", 1,
                 expect=[{"name": "apple", "price": 1.5, "quantity": 1},
                         {"name": "pear", "price": 2.0, "quantity": 3}]),
             _ex(CART, "kiwi", 1, expect=CART))),
    Problem("add_item", "hard", "add_item",
            "Write a Python function add_item(cart, name, price, quantity=1). cart is a list of dicts with keys "
            "name, price, quantity. Return a NEW list (do not modify cart). If an item with the same name is "
            "already in the cart, increase its quantity by quantity; otherwise append a new item. Raise "
            "ValueError if price < 0 or quantity <= 0.",
            (_ex([], "apple", 1.5, expect=CART),
             _ex(CART, "apple", 1.5, 2, expect=[{"name": "apple", "price": 1.5, "quantity": 3}]),
             _ex(CART, "pear", 2.0, expect=CART + [{"name": "pear", "price": 2.0, "quantity": 1}]),
             _ex([], "apple", -1.0, raises="ValueError"),
             _ex([], "apple", 1.5, quantity=0, raises="ValueError"))),
)

SYSTEM = "You are a helpful programmer. Reply with the Python code only."
FENCE = re.compile(r"```[\w+.\-]*\n(.*?)```", re.DOTALL)


def extract_code(reply: str) -> str:
    """The first fenced block if there is one, otherwise the text itself when it looks like code."""
    m = FENCE.search(reply)
    if m:
        return m.group(1)
    unclosed = re.search(r"```[\w+.\-]*\n(.*)", reply, re.DOTALL)
    return unclosed.group(1) if unclosed else reply


def grade(problem: Problem, reply: str, *, timeout: float = 20.0) -> tuple[bool, str]:
    """Run the reply's code against the problem's examples. Returns (passed, short reason)."""
    code = extract_code(reply)
    if f"def {problem.function}" not in code:
        return False, f"no `def {problem.function}` in the reply"
    step = Step("bench", "bench", "bench", ("test_sol.py",), ("sol.py",), (),
                f"sol:{problem.function}", problem.examples)
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "sol.py").write_text(code, encoding="utf-8")
        (Path(d) / "test_sol.py").write_text(render_tests(step), encoding="utf-8")
        res = run_command("python -m pytest -q -p no:cacheprovider", d, allowed=["python", "python3", "pytest"],
                          timeout=timeout, mem_mb=1024, max_output_chars=2000)
    if res.ok:
        return True, "passed"
    if res.timed_out:
        return False, "timed out"
    lines = [l for l in res.output.splitlines() if l.strip()]
    reason = next((l for l in lines if l.startswith(("E ", "E\t")) or "Error" in l), lines[-1] if lines else "failed")
    return False, reason.strip()[:110]


def verdict(scores: dict[str, tuple[int, int]]) -> list[str]:
    """Plain-language reading of the results, by level."""
    def rate(level):
        got = [scores[p.key] for p in PROBLEMS if p.level == level and p.key in scores]
        total = sum(t for _, t in got)
        return sum(g for g, _ in got) / total if total else None
    basic, medium, hard = rate("basic"), rate("medium"), rate("hard")
    out = []
    if basic is None:
        return ["nothing was graded"]
    if basic < 0.5:
        out.append("The model struggles even with tiny functions. It is too weak for code: try a coder model "
                   "(for example qwen2.5-coder:1.5b) or a larger one. No harness can fix this.")
    elif medium is not None and medium < 0.5:
        out.append("The model writes tiny functions but breaks on anything with a loop or a list of dicts. "
                   "Keep every step as small as the 'basic' problems and write the algorithm in the description.")
    elif hard is not None and hard < 0.5:
        out.append("The model handles small and medium functions but not a function that combines several "
                   "rules. Split big functions into small steps (see tests/examples/cart_split.json).")
    else:
        out.append("The model is capable. If the agent still fails, the plan (description/examples) or the "
                   "harness is the problem, not the model.")
    return out


def run_bench(cfg, client, *, tries: int, temperature: float, only: list[str] | None, out,
              problems: tuple[Problem, ...] = PROBLEMS) -> dict[str, tuple[int, int]]:
    chosen = [p for p in problems if not only or p.key in only]
    scores: dict[str, tuple[int, int]] = {}
    out(f"model: {cfg.model}   tries per problem: {tries}   temperature: {temperature}")
    for p in chosen:
        out(f"\n[{p.level}] {p.key}")
        good, errors = 0, 0
        for i in range(1, tries + 1):
            started = time.monotonic()
            try:
                reply = str(client.chat([{"role": "system", "content": SYSTEM},
                                         {"role": "user", "content": p.task}], temperature=temperature))
            except LLMError as e:
                errors += 1
                out(f"  try {i}: model error: {e}")
                continue
            ok, why = grade(p, reply)
            good += ok
            out(f"  try {i}: {'PASS' if ok else 'fail'}  ({time.monotonic() - started:.0f}s)" + ("" if ok else f"  {why}"))
        scores[p.key] = (good, tries - errors)
    out("\n==== results ====")
    for p in chosen:
        g, t = scores[p.key]
        out(f"  {p.level:<6} {p.key:<15} {g}/{t}")
    for line in verdict(scores):
        out("\n" + line)
    return scores


def main(argv=None, *, llm_factory=make_client, out=print) -> int:
    ap = argparse.ArgumentParser(description="Raw model test: can the model write small functions at all?")
    ap.add_argument("--tries", type=int, default=5, help="samples per problem (default 5)")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--only", nargs="+", metavar="PROBLEM")
    args = ap.parse_args(argv)
    known = {p.key for p in PROBLEMS}
    unknown = [k for k in (args.only or []) if k not in known]
    if unknown:
        out(f"unknown problem(s): {', '.join(unknown)}. Choose from: {', '.join(sorted(known))}")
        return 2
    cfg = load_config()
    run_bench(cfg, llm_factory(cfg), tries=args.tries, temperature=args.temperature, only=args.only, out=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
