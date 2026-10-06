"""Talk to the local model server (Ollama, or any OpenAI-compatible server such
as llama.cpp). Standard library only. Refuses non-local URLs unless explicitly
allowed, so code can never leave the machine by accident."""
import ipaddress
import json
import socket
import time
import urllib.error
import urllib.request
from typing import Callable, Protocol
from urllib.parse import urlparse

from codeagent.config import Config
from codeagent.parsing import strip_thinking


class LLMError(Exception):
    pass


class LLMClient(Protocol):
    def chat(self, messages: list[dict]) -> str: ...


def _check_url(cfg: Config) -> str:
    url = cfg.base_url.strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise LLMError(f"invalid URL {cfg.base_url!r} in base_url: expected http://host:port")
    if not cfg.allow_remote_llm:
        host = parsed.hostname
        try:
            is_local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            is_local = host.lower() == "localhost"
        if not is_local:
            raise LLMError(
                f"base_url {cfg.base_url!r} is not a local address. codeagent is offline-only by "
                "default; set allow_remote_llm = true in the config if you really want this."
            )
    return url


class _BaseClient:
    def __init__(self, cfg: Config, *, sleep: Callable[[float], None] = time.sleep, retries: int = 2):
        self.cfg = cfg
        self.base = _check_url(cfg)
        self.sleep = sleep
        self.retries = retries

    def _request(self, path: str, payload: dict | None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            self.base + path, data=data, method="POST" if payload is not None else "GET",
            headers={"Content-Type": "application/json"},
        )
        last: LLMError | None = None
        for attempt in range(self.retries + 1):
            if attempt:
                self.sleep(2.0 * attempt)                 # backoff: 2s, 4s, ...
            try:
                with urllib.request.urlopen(req, timeout=self.cfg.request_timeout) as resp:
                    raw = resp.read()
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:300]
                if e.code == 404 and "model" in body.lower():
                    raise LLMError(
                        f"model {self.cfg.model!r} not found on the server. Run "
                        "`codeagent doctor` to list installed models, or `ollama pull <name>`."
                    ) from e
                last = LLMError(f"model server returned HTTP {e.code}: {body}")
                if e.code >= 500:
                    continue                              # server busy/loading: retry
                raise last from e
            except (socket.timeout, TimeoutError) as e:
                raise LLMError(f"request timed out after {self.cfg.request_timeout:g}s") from e
            except urllib.error.URLError as e:
                if isinstance(e.reason, (socket.timeout, TimeoutError)):
                    raise LLMError(f"request timed out after {self.cfg.request_timeout:g}s") from e
                last = LLMError(
                    f"cannot reach the model server at {self.base} ({e.reason}). "
                    "Is it running? For Ollama: `ollama serve`."
                )
                continue
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                raise LLMError(f"model server sent invalid JSON: {raw[:200]!r}") from e
        raise last  # type: ignore[misc]

    @staticmethod
    def _finish(text) -> str:
        if not isinstance(text, str):
            raise LLMError("unexpected response shape from the model server")
        text = strip_thinking(text)
        if not text.strip():
            raise LLMError("model returned an empty reply")
        return text


class OllamaClient(_BaseClient):
    def chat(self, messages: list[dict]) -> str:
        c = self.cfg
        data = self._request("/api/chat", {
            "model": c.model, "messages": messages, "stream": False,
            "think": False,                                # tiny models: skip slow hidden reasoning
            "keep_alive": c.keep_alive,
            "options": {"num_ctx": c.num_ctx, "num_predict": c.num_predict, "temperature": c.temperature},
        })
        try:
            return self._finish(data["message"]["content"])
        except (KeyError, TypeError) as e:
            raise LLMError("unexpected response shape from Ollama") from e


class OpenAIClient(_BaseClient):
    def chat(self, messages: list[dict]) -> str:
        c = self.cfg
        data = self._request("/v1/chat/completions", {
            "model": c.model, "messages": messages, "stream": False,
            "max_tokens": c.num_predict, "temperature": c.temperature,
        })
        try:
            return self._finish(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError("unexpected response shape from the server") from e


def make_client(cfg: Config, **kw) -> LLMClient:
    cls = OllamaClient if cfg.backend == "ollama" else OpenAIClient
    return cls(cfg, **kw)


def list_models(cfg: Config) -> list[str]:
    client = OllamaClient(cfg, retries=0) if cfg.backend == "ollama" else OpenAIClient(cfg, retries=0)
    if cfg.backend == "ollama":
        data = client._request("/api/tags", None)
        return [m["name"] for m in data.get("models", [])]
    data = client._request("/v1/models", None)
    return [m["id"] for m in data.get("data", [])]
