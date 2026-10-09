import pytest

from agent import Config, ConfigError, load_config


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    import os
    for k in [k for k in os.environ if k.startswith("CODEAGENT_")]:
        monkeypatch.delenv(k)


def test_defaults_are_safe_for_a_small_laptop():
    c = Config()
    assert (c.backend, c.base_url) == ("ollama", "http://localhost:11434")
    assert c.num_ctx <= 16384 and c.min_free_mem_mb > 0 and c.max_temp_c < 100
    assert c.allow_remote_llm is False and c.stop_on_failure is True
    assert set(c.allowed_commands) == {"python", "python3", "pytest"}


def test_agentic_coding_thinks_by_default():
    c = Config()
    assert c.think is True and c.think_level in ("low", "medium", "high")
    assert c.num_predict_tests <= c.num_predict and c.num_predict_body <= c.num_predict


def test_no_env_means_defaults():
    assert load_config() == Config()


def test_env_overrides_any_setting_by_type(monkeypatch):
    monkeypatch.setenv("CODEAGENT_MODEL", "tiny:1b")
    monkeypatch.setenv("CODEAGENT_THINK", "false")
    monkeypatch.setenv("CODEAGENT_MAX_ATTEMPTS", "7")
    monkeypatch.setenv("CODEAGENT_TEMPERATURE", "0.5")
    monkeypatch.setenv("CODEAGENT_ALLOWED_COMMANDS", "python, make")
    c = load_config()
    assert (c.model, c.think, c.max_attempts, c.temperature) == ("tiny:1b", False, 7, 0.5)
    assert c.allowed_commands == ["python", "make"]


@pytest.mark.parametrize("var, value", [("CODEAGENT_MAX_ATTEMPTS", "many"), ("CODEAGENT_THINK", "maybe"),
                                         ("CODEAGENT_BACKEND", "x"), ("CODEAGENT_THINK_LEVEL", "extreme")])
def test_bad_values_are_clean_errors(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    with pytest.raises(ConfigError):
        load_config()
