import pytest

from codeagent.config import Config, ConfigError, load_config


def test_defaults_are_safe_for_a_small_laptop():
    c = Config()
    assert c.backend == "ollama"
    assert c.base_url == "http://localhost:11434"
    assert c.num_ctx <= 8192          # small context: 2GB VRAM
    assert c.max_attempts >= 1
    assert c.min_free_mem_mb > 0
    assert c.max_temp_c < 100
    assert c.stop_on_failure is True
    assert c.allow_remote_llm is False      # offline by default
    assert set(c.allowed_commands) == {"python", "python3", "pytest"}


def test_load_without_file_returns_defaults(monkeypatch):
    monkeypatch.delenv("CODEAGENT_MODEL", raising=False)
    monkeypatch.delenv("CODEAGENT_BASE_URL", raising=False)
    assert load_config(None) == Config()


def test_load_overrides_from_toml(tmp_path, monkeypatch):
    monkeypatch.delenv("CODEAGENT_MODEL", raising=False)
    f = tmp_path / "c.toml"
    f.write_text('model = "tiny:1b"\nnum_ctx = 2048\ntemperature = 0\nstop_on_failure = false\n')
    c = load_config(f)
    assert c.model == "tiny:1b"
    assert c.num_ctx == 2048
    assert c.temperature == 0.0
    assert c.stop_on_failure is False
    assert c.max_attempts == Config().max_attempts   # untouched keys keep defaults


def test_unknown_key_is_an_error_not_silently_ignored(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text("num_ctxx = 2048\n")
    with pytest.raises(ConfigError, match="num_ctxx"):
        load_config(f)


@pytest.mark.parametrize(
    "line",
    ['num_ctx = "big"', "model = 5", "stop_on_failure = 1", 'allowed_commands = "python"',
     "allowed_commands = [1, 2]"],
)
def test_wrong_types_are_rejected(tmp_path, line):
    f = tmp_path / "c.toml"
    f.write_text(line + "\n")
    with pytest.raises(ConfigError):
        load_config(f)


@pytest.mark.parametrize(
    "line",
    ['backend = "claude"', "max_attempts = 0", "num_ctx = 100", "temperature = 3.0",
     "test_timeout = 0", "min_free_mem_mb = -1"],
)
def test_out_of_range_values_are_rejected(tmp_path, line):
    f = tmp_path / "c.toml"
    f.write_text(line + "\n")
    with pytest.raises(ConfigError):
        load_config(f)


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_invalid_toml_is_an_error(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text("this is = = not toml")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(f)


def test_env_overrides_model_and_base_url(tmp_path, monkeypatch):
    f = tmp_path / "c.toml"
    f.write_text('model = "from-file"\n')
    monkeypatch.setenv("CODEAGENT_MODEL", "from-env")
    monkeypatch.setenv("CODEAGENT_BASE_URL", "http://127.0.0.1:9999")
    c = load_config(f)
    assert c.model == "from-env"
    assert c.base_url == "http://127.0.0.1:9999"
