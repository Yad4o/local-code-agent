"""Configuration: typed defaults + optional TOML file + env overrides.

Defaults target a small laptop (2 GB GPU, ~12 GB RAM). Unknown keys and wrong
types are errors so a typo in the config can never silently do nothing.
"""
import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


class ConfigError(Exception):
    pass


@dataclass
class Config:
    # --- model backend ---
    backend: str = "ollama"                     # "ollama" or "openai" (llama.cpp server etc.)
    base_url: str = "http://localhost:11434"
    model: str = "qwen3.5:0.8b"                 # run `codeagent doctor` to list installed models
    num_ctx: int = 4096                         # context window; keep small for 2 GB VRAM
    num_predict: int = 1536                     # max tokens generated per call
    temperature: float = 0.2
    keep_alive: str = "30m"                     # keep model loaded between calls
    request_timeout: float = 300.0
    allow_remote_llm: bool = False              # False = refuse any non-local base_url (offline guarantee)
    # --- agent loop ---
    max_attempts: int = 4                       # tries per phase (tests / implementation)
    stop_on_failure: bool = True
    step_cooldown_s: float = 2.0                # pause between model calls (laptop heat)
    # --- command / file limits ---
    allowed_commands: list[str] = field(default_factory=lambda: ["python", "python3", "pytest"])
    test_timeout: float = 120.0
    command_mem_mb: int = 2048                  # address-space cap for spawned commands
    max_output_chars: int = 3000                # tail of command output fed back to the model
    max_file_bytes: int = 100_000
    max_context_chars: int = 6000               # budget for code shown to the model
    # --- resource guard ---
    min_free_mem_mb: int = 1500
    min_battery_pct: int = 20                   # only enforced when running on battery
    max_temp_c: float = 88.0
    min_free_disk_mb: int = 1024
    resource_wait_s: float = 600.0              # how long to wait for resources before pausing
    # --- daemon ---
    poll_seconds: float = 10.0


_RANGES: dict[str, tuple[float, float]] = {
    "num_ctx": (512, 131072),
    "num_predict": (16, 32768),
    "temperature": (0.0, 2.0),
    "request_timeout": (1, 86400),
    "max_attempts": (1, 20),
    "step_cooldown_s": (0, 3600),
    "test_timeout": (1, 86400),
    "command_mem_mb": (64, 1_000_000),
    "max_output_chars": (100, 100_000),
    "max_file_bytes": (100, 10_000_000),
    "max_context_chars": (500, 200_000),
    "min_free_mem_mb": (0, 1_000_000),
    "min_battery_pct": (0, 100),
    "max_temp_c": (30, 120),
    "min_free_disk_mb": (0, 10_000_000),
    "resource_wait_s": (0, 86400 * 7),
    "poll_seconds": (0.1, 86400),
}


def _check(name: str, value, default):
    """Validate type (by the default's type) and range."""
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"{name}: expected true/false, got {value!r}")
    elif isinstance(default, str):
        if not isinstance(value, str):
            raise ConfigError(f"{name}: expected a string, got {value!r}")
    elif isinstance(default, list):
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ConfigError(f"{name}: expected a list of strings, got {value!r}")
    elif isinstance(default, (int, float)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{name}: expected a number, got {value!r}")
        if isinstance(default, int) and isinstance(value, float) and not value.is_integer():
            raise ConfigError(f"{name}: expected a whole number, got {value!r}")
        lo, hi = _RANGES[name]
        if not lo <= value <= hi:
            raise ConfigError(f"{name}: {value!r} is outside the allowed range {lo}..{hi}")
        value = type(default)(value)
    return value


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    defaults = {f.name: getattr(cfg, f.name) for f in fields(Config)}

    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        try:
            data = tomllib.loads(p.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"invalid TOML in {p}: {e}") from e
        unknown = sorted(set(data) - set(defaults))
        if unknown:
            raise ConfigError(f"unknown config key(s): {', '.join(unknown)}")
        for name, value in data.items():
            setattr(cfg, name, _check(name, value, defaults[name]))

    if os.environ.get("CODEAGENT_MODEL"):
        cfg.model = os.environ["CODEAGENT_MODEL"]
    if os.environ.get("CODEAGENT_BASE_URL"):
        cfg.base_url = os.environ["CODEAGENT_BASE_URL"]

    if cfg.backend not in ("ollama", "openai"):
        raise ConfigError(f"backend: must be 'ollama' or 'openai', got {cfg.backend!r}")
    return cfg
