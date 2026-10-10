import pytest

from agent import build_impl_prompt, load_config, signature_view
from agent import ConfigError
from tests.test_body_hints import make_plan

DATES = "import datetime\n\ndef validate_date(date_text):\n    parsed = datetime.date.fromisoformat(date_text)\n    return True\n"
MERGE = "def merge_quantity(cart, name, quantity=1):\n    result = []\n    return result\n"
CHECKS = 'def check_args(price, quantity):\n    """Raise ValueError for bad input.\n\n    More text."""\n    if price < 0:\n        raise ValueError\n    return True\n'


def test_signature_view_keeps_only_the_def_line_and_the_first_docstring_line():
    assert signature_view(DATES) == "def validate_date(date_text):\n    ..."
    assert signature_view(CHECKS) == 'def check_args(price, quantity):\n    """Raise ValueError for bad input."""\n    ...'
    assert "result" not in signature_view(MERGE) and "quantity=1" in signature_view(MERGE)


def test_signature_view_lists_every_function_and_returns_none_without_one():
    assert signature_view(DATES + "\n\ndef other(x, *, y=2) -> int:\n    return x\n").count("def ") == 2
    assert signature_view("X = 1\n") is None
    assert signature_view("def broken(:") is None


def _user(**kw):
    step = make_plan().steps[0]
    existing = {"cart/merge.py": MERGE, "cart/checks.py": CHECKS}
    return build_impl_prompt(make_plan(), step, {}, existing, 6000, None, None, **kw)[1]["content"]


def test_helper_bodies_are_hidden_by_default_so_the_model_calls_them_instead_of_pasting_them():
    user = _user()
    assert "def merge_quantity(cart, name, quantity=1):" in user and "def check_args(price, quantity):" in user
    assert "result = []" not in user and "raise ValueError" not in user.split("Reply with ONLY")[0].split("tests check")[-1]
    assert "Functions you can call" in user and "do not rewrite their code" in user
    assert "Already imported for you: merge_quantity, check_args" in user


def test_the_full_view_still_shows_the_whole_helper_file():
    user = _user(context_view="full")
    assert "result = []" in user and "Related files (read-only):" in user


def test_a_context_file_without_functions_is_shown_whole():
    step = make_plan().steps[0]
    user = build_impl_prompt(make_plan(), step, {}, {"cart/merge.py": "RATE = 5\n", "cart/checks.py": None},
                             6000, None, None)[1]["content"]
    assert "RATE = 5" in user


def test_context_view_is_validated_and_can_be_set_from_the_environment(monkeypatch):
    monkeypatch.setenv("CODEAGENT_CONTEXT_VIEW", "full")
    assert load_config().context_view == "full"
    monkeypatch.setenv("CODEAGENT_CONTEXT_VIEW", "bodies")
    with pytest.raises(ConfigError):
        load_config()
