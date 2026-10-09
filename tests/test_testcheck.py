from agent import check_test_source, diagnose_broken_tests


def test_the_exact_file_from_the_failed_run_is_rejected():
    src = ("from slugify.core import slugify\n\n"
           "def test_a(text: str) -> None:\n    assert slugify(text) == 'x'\n\n"
           "def test_b(text: str) -> None:\n    assert slugify(text) == 'y'\n")
    problems = check_test_source(src)
    assert len(problems) == 2 and "text" in problems[0] and "NO parameters" in problems[0]


def test_plain_tests_pass():
    assert check_test_source("def test_a():\n    assert 1 == 1\n") == []


def test_parametrize_supplies_the_parameters():
    src = ("import pytest\n\n@pytest.mark.parametrize('text, out', [('A', 'a')])\n"
           "def test_a(text, out):\n    assert text.lower() == out\n")
    assert check_test_source(src) == []


def test_parametrize_with_a_list_of_names():
    src = ("import pytest\n\n@pytest.mark.parametrize(['text', 'out'], [('A', 'a')])\n"
           "def test_a(text, out):\n    assert text.lower() == out\n")
    assert check_test_source(src) == []


def test_parametrize_does_not_cover_a_different_parameter():
    src = ("import pytest\n\n@pytest.mark.parametrize('text', ['A'])\n"
           "def test_a(text, other):\n    assert text\n")
    assert "other" in check_test_source(src)[0]


def test_builtin_fixtures_are_fine():
    assert check_test_source("def test_a(tmp_path, monkeypatch):\n    assert tmp_path\n") == []


def test_fixture_defined_in_the_same_file_is_fine_and_is_not_itself_a_test():
    src = ("import pytest\n\n@pytest.fixture\ndef text():\n    return 'A'\n\n"
           "def test_a(text):\n    assert text == 'A'\n")
    assert check_test_source(src) == []


def test_fixture_from_conftest_is_fine_when_known():
    src = "def test_a(stub):\n    assert stub\n"
    assert check_test_source(src)
    assert check_test_source(src, known_fixtures={"stub"}) == []


def test_parameters_with_defaults_are_fine():
    assert check_test_source("def test_a(x=1):\n    assert x\n") == []


def test_methods_in_a_test_class_ignore_self():
    assert check_test_source("class TestA:\n    def test_a(self):\n        assert True\n") == []


def test_file_with_no_tests_is_rejected():
    assert "no test function" in check_test_source("def helper():\n    return 1\n")[0]


def test_syntax_errors_are_left_to_the_runner():
    assert check_test_source("def test_a(:\n") == []


def test_diagnose_missing_fixture():
    out = "E       fixture 'text' not found\n11 errors in 0.03s"
    assert "'text'" in diagnose_broken_tests(out)


def test_diagnose_syntax_error_and_no_tests():
    assert "syntax" in diagnose_broken_tests("E     SyntaxError: invalid syntax\n1 error")
    assert "no tests" in diagnose_broken_tests("no tests ran in 0.01s")


def test_a_missing_module_is_the_expected_red_not_a_broken_test():
    out = ("E   ModuleNotFoundError: No module named 'slugify'\n"
           "ERROR tests/test_core.py\n1 error in 0.03s")
    assert diagnose_broken_tests(out) is None


def test_a_real_assertion_failure_is_not_diagnosed_as_broken():
    assert diagnose_broken_tests("E   assert '' == 'hello-world'\n1 failed in 0.02s") is None


def test_duplicate_test_names_are_rejected():
    src = "def test_a():\n    assert 1\n\ndef test_a():\n    assert 2\n"
    assert "defined twice" in check_test_source(src)[0]


def test_same_name_in_different_classes_is_fine():
    src = "class A:\n    def test_x(self):\n        assert 1\n\nclass B:\n    def test_x(self):\n        assert 1\n"
    assert check_test_source(src) == []
