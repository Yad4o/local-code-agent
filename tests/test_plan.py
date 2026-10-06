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
