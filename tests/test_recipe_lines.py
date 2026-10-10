from agent import Config, Runner, Workspace, missing_recipe_lines, parse_plan
from tests.test_runner import BODY_OK, FakeGuard, FakeLLM, block, examples_plan

CATEGORY = ('Create normalize_category(category) in expense_manager/categories.py.\n'
            '1. category = category.strip().lower()\n2. if category == "": raise ValueError\n3. return category\n')
REMOVE = ('1. result = []\n2. For each task in tasks:\n   - if task["id"] != task_id:\n     - result.append(task)\n'
          '3. if len(result) == len(tasks): raise KeyError\n4. return result\n')
SUMMARY = ('1. done = 0\n2. For each task in tasks:\n   - if task["done"]: done = done + 1\n'
           '3. return str(len(tasks) - done) + " open, " + str(done) + " done"\n')


def test_a_body_that_folds_the_first_assignment_into_the_return_is_reported():
    # real run: the model moved .strip().lower() into the return, so the empty check saw the untrimmed text
    body = "if category == '':\n    raise ValueError\nreturn category.strip().lower()"
    assert missing_recipe_lines(CATEGORY, body) == ["category = category.strip().lower()", "return category"]


def test_a_dropped_raise_is_reported():
    body = "result = []\nfor task in tasks:\n    if task['id'] != task_id:\n        result.append(task)\nreturn result"
    assert missing_recipe_lines(REMOVE, body) == ["if len(result) == len(tasks): raise KeyError"]


def test_a_rewritten_return_expression_is_reported():
    body = "done = 0\nfor task in tasks:\n    if task['done']:\n        done = done + 1\nreturn f'{len(tasks) - done} open, {done}'"
    assert missing_recipe_lines(SUMMARY, body) == ['return str(len(tasks) - done) + " open, " + str(done) + " done"']


def test_a_body_that_follows_the_recipe_has_nothing_missing_whatever_the_quotes_and_layout():
    body = ("result = []\nfor task in tasks:\n    if task['id'] != task_id:\n        result.append(task)\n"
            "if len(result) == len(tasks):\n    raise KeyError\nreturn result")
    assert missing_recipe_lines(REMOVE, body) == []


def test_english_lines_and_text_that_does_not_parse_never_count_as_missing():
    assert missing_recipe_lines("Create f(x).\n1. Call check_args(price, quantity).\n2. return x", "return x") == []
    assert missing_recipe_lines(REMOVE, "def broken(:") == []


def test_at_most_three_lines_are_reported():
    desc = "".join(f"{i}. v{i} = {i}\n" for i in range(1, 8))
    assert len(missing_recipe_lines(desc, "pass")) == 3


def test_the_retry_prompt_names_the_missing_line(tmp_path):
    desc = "Create slugify(text) in slugify/core.py.\n1. cleaned = text.strip().lower()\n2. return cleaned"
    plan = parse_plan(examples_plan(signature="slugify(text)", description=desc))
    cfg = Config()
    wrong = block("slugify/core.py", "return text")
    llm = FakeLLM(wrong, BODY_OK)
    Runner(cfg, plan, Workspace(tmp_path / "proj", cfg.max_file_bytes), llm, guard=FakeGuard(),
           log=lambda m: None, sleep=lambda s: None, should_stop=lambda: False).run()
    assert "cleaned = text.strip().lower()" in str(llm.calls[1][-1]["content"])
    assert "missing from your body" in str(llm.calls[1][-1]["content"])


def test_import_lines_are_never_reported_because_they_are_moved_to_the_top_of_the_module():
    desc = "1. import re, decimal\n2. if re.fullmatch(r'a', text) is None: raise ValueError\n3. return 1"
    body = "if re.fullmatch(r'a', text) is None:\n    raise ValueError\nreturn 1"
    assert missing_recipe_lines(desc, body) == []
