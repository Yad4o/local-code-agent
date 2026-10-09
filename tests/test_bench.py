import pytest

from codeagent.bench import PROBLEMS, Problem, extract_code, grade, run_bench, verdict
from codeagent.cli import main
from codeagent.config import Config

REFERENCE = {
    "add": "def add(a, b):\n    return a + b\n",
    "is_even": "def is_even(n):\n    return n % 2 == 0\n",
    "slugify": "import re\n\ndef slugify(text):\n    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')\n",
    "fizzbuzz": ("def fizzbuzz(n):\n    out = []\n    for i in range(1, n + 1):\n"
                 "        out.append('FizzBuzz' if i % 15 == 0 else 'Fizz' if i % 3 == 0 else 'Buzz' if i % 5 == 0 else str(i))\n"
                 "    return out\n"),
    "count_words": ("def count_words(text):\n    out = {}\n    for w in text.lower().split():\n"
                    "        out[w] = out.get(w, 0) + 1\n    return out\n"),
    "merge_quantity": ("def merge_quantity(cart, name, quantity):\n    return [{**i, 'quantity': i['quantity'] + quantity} "
                       "if i['name'] == name else dict(i) for i in cart]\n"),
    "add_item": ("def add_item(cart, name, price, quantity=1):\n    if price < 0 or quantity <= 0:\n        raise ValueError('bad')\n"
                 "    out = [dict(i) for i in cart]\n    for i in out:\n        if i['name'] == name:\n"
                 "            i['quantity'] += quantity\n            return out\n"
                 "    return out + [{'name': name, 'price': price, 'quantity': quantity}]\n"),
}


@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.key)
def test_every_problem_is_solvable_so_a_failure_means_the_model_not_the_problem(problem):
    ok, why = grade(problem, f"```python\n{REFERENCE[problem.key]}```")
    assert ok, why


def test_there_is_a_reference_for_every_problem_and_levels_are_known():
    assert set(REFERENCE) == {p.key for p in PROBLEMS}
    assert {p.level for p in PROBLEMS} == {"basic", "medium", "hard"}


def test_extract_code_variants():
    assert extract_code("sure:\n```python\nx = 1\n```\nbye") == "x = 1\n"
    assert extract_code("```\ny = 2\n```") == "y = 2\n"
    assert extract_code("```python\nz = 3\n") == "z = 3\n"          # forgot the closing fence
    assert extract_code("def f(): pass") == "def f(): pass"


def test_wrong_code_fails_with_a_reason():
    p = next(p for p in PROBLEMS if p.key == "add")
    ok, why = grade(p, "```python\ndef add(a, b):\n    return a - b\n```")
    assert not ok and "returned" in why


def test_reply_without_the_function_fails():
    p = next(p for p in PROBLEMS if p.key == "add")
    assert grade(p, "I would use the + operator.") == (False, "no `def add` in the reply")


def test_mutating_the_input_and_returning_none_fails_the_hard_problem():
    p = next(p for p in PROBLEMS if p.key == "add_item")
    bad = "def add_item(cart, name, price, quantity=1):\n    cart.append({'name': name})\n"
    assert grade(p, f"```python\n{bad}```")[0] is False


def test_infinite_loop_times_out_instead_of_hanging():
    p = next(p for p in PROBLEMS if p.key == "add")
    ok, why = grade(p, "def add(a, b):\n    while True:\n        pass\n", timeout=2)
    assert not ok and why == "timed out"


class Scripted:
    """Answers each problem from a table: reference code, or something broken."""
    def __init__(self, good):
        self.good, self.n = good, 0

    def chat(self, messages, **kw):
        self.n += 1
        task = messages[-1]["content"]
        for p in PROBLEMS:
            if f"function {p.function}(" in task:
                return f"```python\n{REFERENCE[p.key]}```" if p.key in self.good else "```python\ndef nope(): pass\n```"
        raise AssertionError(task)


def test_run_bench_scores_each_problem_and_prints_a_verdict():
    lines = []
    client = Scripted(good={"add", "is_even", "slugify", "fizzbuzz", "count_words", "merge_quantity"})
    scores = run_bench(Config(), client, tries=2, temperature=0.4, only=None, out=lines.append)
    assert scores["add"] == (2, 2) and scores["add_item"] == (0, 2)
    text = "\n".join(lines)
    import re
    assert re.search(r"hard\s+add_item\s+0/2", text) and "Split big functions" in text


def test_only_filters_problems():
    lines = []
    scores = run_bench(Config(), Scripted({"add"}), tries=1, temperature=0.2, only=["add"], out=lines.append)
    assert list(scores) == ["add"]


def test_model_errors_are_counted_not_scored_as_failures():
    from codeagent.llm import LLMError

    class Down:
        def chat(self, *a, **k):
            raise LLMError("server down")
    lines = []
    scores = run_bench(Config(), Down(), tries=2, temperature=0.4, only=["add"], out=lines.append)
    assert scores["add"] == (0, 0) and any("model error" in l for l in lines)


def test_verdict_levels():
    base = {p.key: (5, 5) for p in PROBLEMS}
    assert "model is capable" in verdict(base)[0]
    assert "Split big functions" in verdict({**base, "add_item": (0, 5)})[0]
    assert "loop or a list of dicts" in verdict({**base, "fizzbuzz": (0, 5), "count_words": (1, 5),
                                                 "merge_quantity": (0, 5), "add_item": (0, 5)})[0]
    assert "too weak for code" in verdict({p.key: (0, 5) for p in PROBLEMS})[0]
    assert verdict({}) == ["nothing was graded"]


def test_cli_bench_runs_and_rejects_unknown_problem(tmp_path):
    lines = []
    code = main(["bench", "--tries", "1", "--only", "add", "is_even"], out=lines.append,
                llm_factory=lambda c: Scripted({"add", "is_even"}))
    assert code == 0 and any("add" in l and "1/1" in l for l in lines)
    bad = []
    assert main(["bench", "--only", "nope"], out=bad.append, llm_factory=lambda c: Scripted(set())) == 2
