import socket

import pytest

from agent import Config, LLMError, list_models, make_client

MSGS = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]


def cfg_for(stub, **kw):
    kw.setdefault("base_url", stub.url)
    return Config(**kw)


def ollama_reply(text):
    return (200, {"message": {"role": "assistant", "content": text}, "done": True})


def openai_reply(text):
    return (200, {"choices": [{"message": {"role": "assistant", "content": text}}]})


# --- ollama ----------------------------------------------------------------

def test_ollama_chat_sends_expected_payload_and_returns_content(stub):
    stub.responses = [ollama_reply("hello there")]
    c = cfg_for(stub, model="m:1b", num_ctx=2048, num_predict=300, temperature=0.1, keep_alive="5m")
    assert make_client(c).chat(MSGS) == "hello there"
    method, path, body = stub.requests[0]
    assert (method, path) == ("POST", "/api/chat")
    assert body["model"] == "m:1b"
    assert body["messages"] == MSGS
    assert body["stream"] is False
    assert body["think"] is True                       # agentic coding: the model reasons before it writes
    assert body["keep_alive"] == "5m"
    assert body["options"] == {"num_ctx": 2048, "num_predict": 300, "temperature": 0.1, "repeat_penalty": 1.1}


def test_thinking_can_be_switched_off_and_gpt_oss_gets_a_level(stub):
    stub.responses = [ollama_reply("a")]
    for kw in ({"think": False}, {"model": "gpt-oss:20b", "think_level": "medium"}, {"model": "gpt-oss:20b", "think": False}):
        make_client(cfg_for(stub, **kw)).chat(MSGS)
    assert [r[2]["think"] for r in stub.requests] == [False, "medium", False]


def test_running_out_of_tokens_while_thinking_is_a_clear_error(stub):
    stub.responses = [(200, {"message": {"content": "", "thinking": "hmm" * 50}, "done_reason": "length"})]
    with pytest.raises(LLMError, match="thinking"):
        make_client(cfg_for(stub, num_predict=64)).chat(MSGS)


def test_stop_sequences_go_to_the_server_only_when_not_thinking(stub):
    stub.responses = [ollama_reply("a"), ollama_reply("b"), ollama_reply("c")]
    make_client(cfg_for(stub, think=False)).chat(MSGS)
    make_client(cfg_for(stub, think=False)).chat(MSGS, stop=["\nFILE:"])
    make_client(cfg_for(stub)).chat(MSGS, stop=["\nFILE:"])
    assert "stop" not in stub.requests[0][2]["options"]
    assert stub.requests[1][2]["options"]["stop"] == ["\nFILE:"]
    assert "stop" not in stub.requests[2][2]["options"]       # the reasoning may mention the marker and end the reply


def test_stop_sequences_are_applied_to_the_answer_when_thinking(stub):
    stub.responses = [ollama_reply("x = 1\nFILE: other.py\nmore")]
    assert make_client(cfg_for(stub)).chat(MSGS, stop=["\nFILE:"]) == "x = 1"


def test_empty_answer_explains_what_the_model_did(stub):
    stub.responses = [(200, {"message": {"content": "", "thinking": "let me think " * 20}, "done_reason": "stop",
                             "eval_count": 77})]
    with pytest.raises(LLMError) as e:
        make_client(cfg_for(stub)).chat(MSGS)
    msg = str(e.value)
    assert "done_reason='stop'" in msg and "77 tokens" in msg and "let me think" in msg and "CODEAGENT_THINK=false" in msg

def test_thinking_text_is_stripped_from_replies(stub):
    stub.responses = [ollama_reply("<think>hmm</think>\nanswer")]
    assert make_client(cfg_for(stub)).chat(MSGS) == "answer"


def test_empty_reply_is_an_error(stub):
    stub.responses = [ollama_reply("   ")]
    with pytest.raises(LLMError, match="no answer"):
        make_client(cfg_for(stub)).chat(MSGS)


def test_non_json_reply_is_an_error(stub):
    stub.responses = [(200, "<html>oops</html>")]
    with pytest.raises(LLMError, match="invalid"):
        make_client(cfg_for(stub)).chat(MSGS)


def test_unexpected_json_shape_is_an_error(stub):
    stub.responses = [(200, {"nothing": "here"})]
    with pytest.raises(LLMError):
        make_client(cfg_for(stub)).chat(MSGS)


def test_model_not_found_message_is_helpful(stub):
    stub.responses = [(404, {"error": "model 'qwen3.5:0.8b' not found"})]
    with pytest.raises(LLMError) as exc:
        make_client(cfg_for(stub, model="qwen3.5:0.8b")).chat(MSGS)
    assert "qwen3.5:0.8b" in str(exc.value)
    assert "ollama pull" in str(exc.value) or "doctor" in str(exc.value)


def test_connection_refused_message_is_helpful():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()                                          # nothing listens here now
    c = Config(base_url=f"http://127.0.0.1:{port}")
    with pytest.raises(LLMError) as exc:
        make_client(c, sleep=lambda s: None).chat(MSGS)
    assert "cannot reach" in str(exc.value)
    assert f"127.0.0.1:{port}" in str(exc.value)


def test_timeout_is_an_error(stub):
    stub.delay = 2.5
    stub.responses = [ollama_reply("late")]
    with pytest.raises(LLMError, match="timed out"):
        make_client(cfg_for(stub, request_timeout=1.0), sleep=lambda s: None, retries=0).chat(MSGS)


# --- retries -----------------------------------------------------------------

def test_server_errors_are_retried_then_succeed(stub):
    stub.responses = [(503, {"error": "loading model"}), (500, "boom"), ollama_reply("ok")]
    sleeps = []
    out = make_client(cfg_for(stub), sleep=sleeps.append, retries=2).chat(MSGS)
    assert out == "ok"
    assert len(stub.requests) == 3
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]       # backoff grows


def test_gives_up_after_retries(stub):
    stub.responses = [(503, {"error": "busy"})]
    with pytest.raises(LLMError, match="503"):
        make_client(cfg_for(stub), sleep=lambda s: None, retries=2).chat(MSGS)
    assert len(stub.requests) == 3


def test_client_errors_are_not_retried(stub):
    stub.responses = [(400, {"error": "bad request"})]
    with pytest.raises(LLMError, match="400"):
        make_client(cfg_for(stub), sleep=lambda s: None, retries=3).chat(MSGS)
    assert len(stub.requests) == 1


# --- openai-compatible (llama.cpp server etc.) ---------------------------------

def test_openai_chat(stub):
    stub.responses = [openai_reply("from llama.cpp")]
    c = cfg_for(stub, backend="openai", model="qwen", num_predict=200, temperature=0.3)
    assert make_client(c).chat(MSGS) == "from llama.cpp"
    method, path, body = stub.requests[0]
    assert (method, path) == ("POST", "/v1/chat/completions")
    assert body["model"] == "qwen" and body["messages"] == MSGS
    assert body["max_tokens"] == 200 and body["temperature"] == 0.3
    assert body["stream"] is False


# --- model listing -------------------------------------------------------------

def test_list_models_ollama(stub):
    stub.responses = [(200, {"models": [{"name": "a:1b"}, {"name": "b:7b"}]})]
    assert list_models(cfg_for(stub)) == ["a:1b", "b:7b"]
    assert stub.requests[0][:2] == ("GET", "/api/tags")


def test_list_models_openai(stub):
    stub.responses = [(200, {"data": [{"id": "x"}, {"id": "y"}]})]
    assert list_models(cfg_for(stub, backend="openai")) == ["x", "y"]
    assert stub.requests[0][:2] == ("GET", "/v1/models")


# --- offline guarantee -----------------------------------------------------------

@pytest.mark.parametrize(
    "url", ["http://localhost:11434", "http://127.0.0.1:8080", "http://[::1]:11434", "http://127.0.0.5:1"],
)
def test_local_urls_are_accepted(url):
    make_client(Config(base_url=url))


@pytest.mark.parametrize(
    "url", ["https://api.openai.com", "http://example.com:11434", "http://8.8.8.8", "http://192.168.1.20:11434",
            "ftp://localhost", "localhost:11434", ""],
)
def test_non_local_or_odd_urls_are_refused(url):
    with pytest.raises(LLMError, match="local|URL"):
        make_client(Config(base_url=url))


def test_remote_allowed_only_with_explicit_flag():
    make_client(Config(base_url="http://192.168.1.20:11434", allow_remote_llm=True))


# --- truncation awareness + per-call overrides -------------------------------------

def test_ollama_length_stop_marks_the_reply_as_truncated(stub):
    stub.responses = [(200, {"message": {"content": "partial"}, "done": True, "done_reason": "length"})]
    r = make_client(cfg_for(stub)).chat(MSGS)
    assert r == "partial" and r.truncated is True


@pytest.mark.parametrize("extra", [{"done_reason": "stop"}, {}])
def test_ollama_normal_stop_is_not_truncated(stub, extra):
    stub.responses = [(200, {"message": {"content": "whole"}, "done": True, **extra})]
    assert make_client(cfg_for(stub)).chat(MSGS).truncated is False


def test_openai_finish_reason_length_is_truncated(stub):
    stub.responses = [(200, {"choices": [{"message": {"content": "cut"}, "finish_reason": "length"}]})]
    assert make_client(cfg_for(stub, backend="openai")).chat(MSGS).truncated is True
    stub.responses = [(200, {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})]
    assert make_client(cfg_for(stub, backend="openai")).chat(MSGS).truncated is False


def test_per_call_temperature_and_token_budget_override_the_config(stub):
    stub.responses = [ollama_reply("x")]
    make_client(cfg_for(stub, temperature=0.2, num_predict=1500)).chat(MSGS, temperature=0.7, num_predict=400)
    opts = stub.requests[0][2]["options"]
    assert opts["temperature"] == 0.7 and opts["num_predict"] == 400


def test_per_call_overrides_for_openai_backend(stub):
    stub.responses = [openai_reply("x")]
    make_client(cfg_for(stub, backend="openai")).chat(MSGS, temperature=0.9, num_predict=300)
    body = stub.requests[0][2]
    assert body["temperature"] == 0.9 and body["max_tokens"] == 300


def test_openai_stop_sequences(stub):
    stub.responses = [openai_reply("x")]
    make_client(cfg_for(stub, backend="openai")).chat(MSGS, stop=["\nFILE:"])
    assert stub.requests[0][2]["stop"] == ["\nFILE:"]
