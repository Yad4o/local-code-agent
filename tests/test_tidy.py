import shutil

import pytest

from agent import ruff_fix, ruff_problems, tidy_python

needs_ruff = pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not installed")


def test_docstrings_and_comments_are_removed_and_code_is_reprinted():
    src = ('"""module doc"""\nimport re\n\ndef f(x):\n    """Does a thing.\n\n    Args: x\n    """\n'
           '    # lower it\n    return re.sub(r"[^a-z]+",   "-", x.lower())   # trailing\n')
    out = tidy_python(src)
    assert "doc" not in out and "#" not in out and "Args" not in out
    assert out == "import re\n\ndef f(x):\n    return re.sub('[^a-z]+', '-', x.lower())\n"


def test_a_function_that_only_had_a_docstring_keeps_a_valid_body():
    out = tidy_python('def f():\n    """only a docstring"""\n')
    assert out == "def f():\n    pass\n"


def test_nested_and_class_docstrings_go_too():
    out = tidy_python('class A:\n    """c"""\n    def m(self):\n        """m"""\n        return 1\n')
    assert '"""' not in out and "'c'" not in out and "return 1" in out


def test_strings_that_are_not_docstrings_stay():
    out = tidy_python("def f():\n    x = 'keep me'\n    return x\n")
    assert "'keep me'" in out


def test_empty_source_and_invalid_source_are_left_alone():
    assert tidy_python("") == ""
    assert tidy_python("def broken(:\n") == "def broken(:\n"


def test_behaviour_is_unchanged_by_tidying():
    src = "def f(a, b=2):\n    '''doc'''\n    return a * b + 1  # c\n"
    ns, ns2 = {}, {}
    exec(src, ns)
    exec(tidy_python(src), ns2)
    assert ns["f"](3) == ns2["f"](3) == 7


@needs_ruff
def test_ruff_removes_unused_imports_but_not_in_init_files(tmp_path):
    (tmp_path / "m.py").write_text("import os\nimport re\n\ndef f():\n    return re\n")
    (tmp_path / "__init__.py").write_text("from .m import f\n")
    ruff_fix(tmp_path, ["m.py", "__init__.py"])
    assert (tmp_path / "m.py").read_text() == "import re\n\ndef f():\n    return re\n"
    assert (tmp_path / "__init__.py").read_text() == "from .m import f\n"


@needs_ruff
def test_ruff_reports_undefined_names_with_line_numbers(tmp_path):
    (tmp_path / "m.py").write_text("def f():\n    return missing_name\n")
    problems = ruff_problems(tmp_path, ["m.py"])
    assert problems and "F821" in problems[0] and "m.py:2" in problems[0]


@needs_ruff
def test_ruff_is_quiet_on_clean_code(tmp_path):
    (tmp_path / "m.py").write_text("def f():\n    return 1\n")
    assert ruff_problems(tmp_path, ["m.py"]) == []


def test_without_ruff_nothing_happens_and_nothing_breaks(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    (tmp_path / "m.py").write_text("import os\nreturn nope\n")
    ruff_fix(tmp_path, ["m.py"])
    assert ruff_problems(tmp_path, ["m.py"]) == []
    assert (tmp_path / "m.py").read_text() == "import os\nreturn nope\n"
