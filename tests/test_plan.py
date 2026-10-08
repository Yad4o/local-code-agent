import pytest

from codeagent.plan import Plan, PlanError, load_plan, parse_plan, plan_slug, template_text

VALID = '''
title = "Slugify utility"
test_command = "python -m pytest -q"

[[step]]
title = "core function"
description = "Create slugify(text) in slugify/core.py that lowercases and dashes words."
commit = "feat: add slugify core function"
tests = ["tests/test_core.py"]
files = ["slugify/__init__.py", "slugify/core.py"]
'''


def errors_of(text):
    with pytest.raises(PlanError) as exc:
        parse_plan(text)
    return exc.value.errors


def replace(old, new):
    assert old in VALID
    return VALID.replace(old, new)


def test_valid_plan_parses():
    p = parse_plan(VALID)
    assert isinstance(p, Plan)
    assert p.title == "Slugify utility"
    assert p.test_command == "python -m pytest -q"
    assert p.workspace is None
    assert len(p.steps) == 1
    s = p.steps[0]
    assert s.title == "core function"
    assert s.commit == "feat: add slugify core function"
    assert s.tests == ("tests/test_core.py",)
    assert s.files == ("slugify/__init__.py", "slugify/core.py")
    assert s.context == ()


def test_optional_workspace_and_context():
    text = 'workspace = "~/p/x"\n' + VALID.replace(
        'files = [', 'context = ["README.md"]\nfiles = [')
    p = parse_plan(text)
    assert p.workspace == "~/p/x"
    assert p.steps[0].context == ("README.md",)


def test_invalid_toml():
    assert "invalid TOML" in errors_of("title = = oops")[0]


@pytest.mark.parametrize("field", ["title", "test_command"])
def test_top_level_required(field):
    text = "\n".join(l for l in VALID.splitlines() if not l.startswith(field + " "))
    assert any(field in e for e in errors_of(text))


def test_requires_at_least_one_step():
    errs = errors_of('title = "x"\ntest_command = "pytest"\n')
    assert any("at least one [[step]]" in e for e in errs)


@pytest.mark.parametrize("field", ["title", "description", "commit", "tests", "files"])
def test_step_required_fields(field):
    text = "\n".join(l for l in VALID.splitlines()
                     if not l.startswith(field + " ") or l.startswith("test_command"))
    assert any(f"step 1" in e and field in e for e in errors_of(text))


def test_unknown_keys_are_errors_so_typos_are_caught():
    assert any("unknown" in e and "titel" in e for e in errors_of('titel = "x"\n' + VALID))
    assert any("unknown" in e and "tests_" in e
               for e in errors_of(replace('tests = [', 'tests_ = ["a.py"]\ntests = [')))
    assert any("unknown" in e and "zzz" in e for e in errors_of('zzz = 1\n' + VALID))


@pytest.mark.parametrize(
    "path", ["/etc/passwd", "../evil.py", "a/../../b.py", ".git/config", ".codeagent/state.json",
             "", "a\\b.py"],
)
def test_unsafe_paths_rejected(path):
    errs = errors_of(replace('files = ["slugify/__init__.py", "slugify/core.py"]',
                             f'files = ["{path}"]'.replace("\\", "\\\\")))
    assert any("path" in e for e in errs)


def test_tests_and_files_must_not_overlap():
    errs = errors_of(replace('tests = ["tests/test_core.py"]', 'tests = ["slugify/core.py"]'))
    assert any("both" in e for e in errs)


def test_duplicate_paths_rejected():
    errs = errors_of(replace('tests = ["tests/test_core.py"]',
                             'tests = ["tests/test_core.py", "tests/test_core.py"]'))
    assert any("duplicate" in e for e in errs)


def test_duplicate_step_titles_rejected():
    errs = errors_of(VALID + VALID.split("[[step]]")[1].join(["[[step]]", ""]))
    assert any("duplicate step title" in e for e in errs)


@pytest.mark.parametrize(
    "commit", ["", "   ", "x" * 73, "line one\\nline two"],
)
def test_bad_commit_messages_rejected(commit):
    errs = errors_of(replace('commit = "feat: add slugify core function"', f'commit = "{commit}"'))
    assert any("commit" in e for e in errs)


def test_all_errors_reported_at_once():
    text = 'title = ""\n[[step]]\ntitle = ""\n'
    errs = errors_of(text)
    assert len(errs) >= 4


def test_error_message_lists_everything():
    with pytest.raises(PlanError) as exc:
        parse_plan('title = ""\n')
    assert str(exc.value).count("\n") >= 1


def test_load_plan_from_file(tmp_path):
    f = tmp_path / "p.toml"
    f.write_text(VALID)
    assert load_plan(f).title == "Slugify utility"


def test_load_plan_missing_file(tmp_path):
    with pytest.raises(PlanError, match="not found"):
        load_plan(tmp_path / "nope.toml")


def test_slug():
    assert plan_slug(parse_plan(VALID)) == "slugify-utility"
    assert plan_slug(Plan("  Hello,   World!! ", "pytest", None, ())) == "hello-world"
    assert plan_slug(Plan("!!!", "pytest", None, ())) == "plan"


def test_bundled_template_is_a_valid_plan():
    p = parse_plan(template_text())
    assert len(p.steps) >= 1


# ----------------------------------------------------------------- function + examples

EX = VALID + '''function = "slugify.core:slugify"

[[step.examples]]
args = ["Hello, World!"]
expect = "hello-world"

[[step.examples]]
args = [123]
raises = "TypeError"
'''


def test_function_and_examples_parse():
    s = parse_plan(EX).steps[0]
    assert s.function == "slugify.core:slugify"
    assert len(s.examples) == 2
    assert s.examples[0].args == ("Hello, World!",) and s.examples[0].expect == "hello-world"
    assert s.examples[1].raises == "TypeError"


def test_a_step_without_examples_is_unchanged():
    s = parse_plan(VALID).steps[0]
    assert s.function is None and s.examples == ()


def test_function_and_examples_must_come_together():
    assert any("go together" in e for e in errors_of(VALID + 'function = "slugify.core:slugify"\n'))
    assert any("go together" in e for e in errors_of(VALID + '[[step.examples]]\nargs = ["a"]\nexpect = "a"\n'))


def test_function_must_be_module_colon_name_and_its_file_listed():
    bad = EX.replace('"slugify.core:slugify"', '"slugify"')
    assert any("package.module:name" in e for e in errors_of(bad))
    other = EX.replace('"slugify.core:slugify"', '"slugify.other:slugify"')
    assert any("slugify/other.py must be listed" in e for e in errors_of(other))


def test_examples_need_exactly_one_tests_file():
    two = EX.replace('tests = ["tests/test_core.py"]', 'tests = ["tests/test_core.py", "tests/test_more.py"]')
    assert any("exactly one path" in e for e in errors_of(two))


def test_example_needs_exactly_one_of_expect_or_raises():
    both = EX + 'expect = 1\n'
    neither = VALID + 'function = "slugify.core:slugify"\n[[step.examples]]\nargs = ["a"]\n'
    assert any("exactly one of" in e for e in errors_of(neither))
    assert any("exactly one of" in e for e in errors_of(both))


def test_raises_must_be_a_builtin_exception():
    assert any("builtin exception" in e for e in errors_of(EX.replace('"TypeError"', '"print"')))
    assert any("builtin exception" in e for e in errors_of(EX.replace('"TypeError"', '"Nope"')))


def test_example_values_must_be_printable_python():
    assert any("nan/inf" in e for e in errors_of(EX.replace('expect = "hello-world"', 'expect = nan')))
    assert any("unsupported type" in e for e in errors_of(EX.replace('expect = "hello-world"', 'expect = 1979-05-27')))


def test_unknown_example_key_and_bad_kwargs_are_reported():
    assert any("unknown key 'result'" in e for e in errors_of(EX + 'result = 1\n'))
    assert any("'kwargs'" in e for e in errors_of(EX.replace('args = [123]', 'args = [123]\nkwargs = { "a b" = 1 }')))
