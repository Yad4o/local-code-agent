import copy
import json

import pytest

from agent import TEMPLATE, Plan, PlanError, load_plan, parse_plan, plan_slug

VALID = {
    "title": "Slugify utility",
    "test_command": "python -m pytest -q",
    "steps": [{
        "title": "core function",
        "description": "Create slugify(text) in slugify/core.py.",
        "commit": "feat: add slugify core function",
        "tests": ["tests/test_core.py"],
        "files": ["slugify/__init__.py", "slugify/core.py"],
    }],
}
EXAMPLES = {"function": "slugify.core:slugify",
            "examples": [{"args": ["Hello, World!"], "expect": "hello-world"},
                         {"args": [123], "raises": "TypeError"}]}


def make(top=None, step=None, drop=(), drop_step=()):
    d = copy.deepcopy(VALID)
    d["steps"][0].update(step or {})
    for k in drop_step:
        d["steps"][0].pop(k, None)
    d.update(top or {})
    for k in drop:
        d.pop(k, None)
    return d


def errors_of(d):
    with pytest.raises(PlanError) as exc:
        parse_plan(d if isinstance(d, str) else json.dumps(d))
    return exc.value.errors


def has(errs, *words):
    return any(all(w in e for w in words) for e in errs)


def test_valid_plan_parses():
    p = parse_plan(json.dumps(VALID))
    assert isinstance(p, Plan) and p.workspace is None
    s = p.steps[0]
    assert s.tests == ("tests/test_core.py",) and s.files == ("slugify/__init__.py", "slugify/core.py")
    assert s.context == () and s.function is None and s.examples == ()


def test_optional_workspace_and_context():
    p = parse_plan(json.dumps(make(top={"workspace": "~/p/x"}, step={"context": ["README.md"]})))
    assert p.workspace == "~/p/x" and p.steps[0].context == ("README.md",)


def test_template_is_a_valid_plan():
    assert parse_plan(TEMPLATE).steps[0].signature == "slugify(text)"


def test_bad_json_and_non_object():
    assert has(errors_of("{oops"), "invalid JSON")
    assert has(errors_of("[]"), "JSON object")


@pytest.mark.parametrize("field", ["title", "test_command"])
def test_top_level_required(field):
    assert has(errors_of(make(drop=[field])), field)


@pytest.mark.parametrize("field", ["title", "description", "commit", "tests", "files"])
def test_step_required_fields(field):
    assert has(errors_of(make(drop_step=[field])), "step 1", field)


def test_requires_at_least_one_step_and_reports_everything_at_once():
    assert has(errors_of(make(top={"steps": []})), "steps")
    assert len(errors_of({"title": "", "steps": [{"title": ""}]})) >= 4


def test_unknown_keys_are_errors_so_typos_are_caught():
    assert has(errors_of(make(top={"titel": "x"})), "unknown", "titel")
    assert has(errors_of(make(step={"tests_": ["a.py"]})), "unknown", "tests_")


@pytest.mark.parametrize("path", ["/etc/passwd", "../evil.py", "a/../../b.py", ".git/config",
                                  ".codeagent/state.json", "", "a\\b.py"])
def test_unsafe_paths_rejected(path):
    assert has(errors_of(make(step={"files": [path]})), "path")


def test_path_lists_reject_overlap_and_duplicates():
    assert has(errors_of(make(step={"tests": ["slugify/core.py"]})), "both")
    assert has(errors_of(make(step={"tests": ["t.py", "t.py"]})), "duplicate")


def test_duplicate_step_titles_rejected():
    d = make()
    d["steps"].append(copy.deepcopy(d["steps"][0]))
    assert has(errors_of(d), "duplicate step title")


@pytest.mark.parametrize("commit", ["", "   ", "x" * 73, "line one\nline two"])
def test_bad_commit_messages_rejected(commit):
    assert has(errors_of(make(step={"commit": commit})), "commit")


def test_load_plan_from_file_and_missing_file(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(json.dumps(VALID))
    assert load_plan(f).title == "Slugify utility"
    with pytest.raises(PlanError, match="not found"):
        load_plan(tmp_path / "nope.json")


def test_slug():
    assert plan_slug(parse_plan(json.dumps(VALID))) == "slugify-utility"
    assert plan_slug(Plan("  Hello,   World!! ", "pytest", None, ())) == "hello-world"
    assert plan_slug(Plan("!!!", "pytest", None, ())) == "plan"


# ----------------------------------------------------------------- function + examples

def ex(**changes):
    return make(step={**copy.deepcopy(EXAMPLES), **changes})


def test_function_and_examples_parse():
    s = parse_plan(json.dumps(ex())).steps[0]
    assert s.function == "slugify.core:slugify" and len(s.examples) == 2
    assert s.examples[0].args == ("Hello, World!",) and s.examples[0].expect == "hello-world"
    assert s.examples[1].raises == "TypeError"


def test_function_and_examples_go_together():
    assert has(errors_of(make(step={"function": "slugify.core:slugify"})), "go together")
    assert has(errors_of(make(step={"examples": EXAMPLES["examples"]})), "go together")


def test_function_must_be_module_colon_name_and_its_file_listed():
    assert has(errors_of(ex(function="slugify")), "package.module:name")
    assert has(errors_of(ex(function="slugify.other:slugify")), "slugify/other.py must be listed")


def test_examples_need_exactly_one_tests_file():
    assert has(errors_of(ex(tests=["tests/a.py", "tests/b.py"])), "exactly one path")


def test_example_needs_exactly_one_of_expect_or_raises():
    both = [{"args": ["a"], "expect": 1, "raises": "TypeError"}]
    neither = [{"args": ["a"]}]
    assert has(errors_of(ex(examples=both)), "exactly one of")
    assert has(errors_of(ex(examples=neither)), "exactly one of")


@pytest.mark.parametrize("name", ["print", "Nope"])
def test_raises_must_be_a_builtin_exception(name):
    assert has(errors_of(ex(examples=[{"args": [1], "raises": name}])), "builtin exception")


def test_example_values_must_be_printable_python():
    assert has(errors_of('{"title":"t","test_command":"pytest","steps":[]}'.replace(
        "[]", json.dumps(ex()["steps"]).replace('"hello-world"', "NaN"))), "nan/inf")


def test_unknown_example_key_and_bad_kwargs_are_reported():
    assert has(errors_of(ex(examples=[{"args": [1], "expect": 1, "result": 1}])), "unknown key 'result'")
    assert has(errors_of(ex(examples=[{"kwargs": {"a b": 1}, "expect": 1}])), "'kwargs'")
