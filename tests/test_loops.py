import json

import pytest

from agent import LLMError, LoopGuard, NoAnswer, make_client
from tests.test_llm import MSGS, cfg_for, ndjson
from tests.test_runner import BODY_OK, FakeLLM, cfg, make_body_runner  # noqa: F401

VARIED = " ".join(f"step {i} computes value {i * 7} then checks {i % 5}" for i in range(300))


def feed_all(guard, kind, text, size=40):
    for i in range(0, len(text), size):
        reason = guard.feed(kind, text[i:i + size])
        if reason:
            return reason
    return None


# --- the guard itself -------------------------------------------------------

def test_small_loop_is_caught():
    assert "repeat" in feed_all(LoopGuard(), "thinking", "Wait, let me check the list again. " * 8)


def test_big_loop_of_a_whole_paragraph_is_caught():
    para = " ".join(f"word{i}" for i in range(70)) + ". "
    assert feed_all(LoopGuard(), "thinking", VARIED[:300] + para * 3) is not None


def test_same_line_coming_back_again_and_again_is_caught():
    text = "".join(f"considering option number {i} in detail\nBut wait, the list is empty so return it\n"
                   for i in range(12))
    assert feed_all(LoopGuard(), "thinking", text) is not None


def test_normal_long_thinking_and_normal_code_are_not_flagged():
    assert feed_all(LoopGuard(), "thinking", VARIED) is None
    assert feed_all(LoopGuard(), "content", "x = 1\nx = 1\nx = 1\n" + "-" * 200 + "\nreturn x\n") is None


def test_thinking_cap_only_applies_before_an_answer_starts():
    assert "without answering" in feed_all(LoopGuard(max_think_chars=500), "thinking", VARIED)
    g = LoopGuard(max_think_chars=500)
    g.feed("content", "def f(): pass")
    assert feed_all(g, "thinking", VARIED[:1500]) is None
    assert feed_all(LoopGuard(max_think_chars=0), "thinking", VARIED) is None      # 0 = unlimited


# --- the client stops a looping stream --------------------------------------

def looping(kind):
    piece = "Wait, I should double check the quantity again. "
    return ndjson(*[{"message": {kind: piece}, "done": False} for _ in range(40)],
                  {"message": {"content": ""}, "done": True, "done_reason": "stop"})


def test_stream_that_loops_while_thinking_raises_no_answer_and_stops_early(stub):
    stub.responses = [looping("thinking")]
    seen = []
    with pytest.raises(NoAnswer, match="repeat"):
        make_client(cfg_for(stub)).chat(MSGS, on_token=lambda k, t: seen.append(t))
    assert 3 <= len(seen) < 40


def test_stream_that_loops_in_the_answer_keeps_what_came_before_as_a_cut_off_reply(stub):
    stub.responses = [ndjson({"message": {"content": "return 1\n"}, "done": False},
                             *[{"message": {"content": "return result_value_again\n"}, "done": False}] * 30,
                             {"message": {"content": ""}, "done": True, "done_reason": "stop"})]
    reply = make_client(cfg_for(stub)).chat(MSGS, on_token=lambda *a: None)
    assert reply.truncated is True and reply.startswith("return 1")


def test_loop_guard_can_be_switched_off(stub):
    stub.responses = [looping("thinking")]
    with pytest.raises(NoAnswer, match="no answer") as e:
        make_client(cfg_for(stub, loop_guard=False)).chat(MSGS, on_token=lambda *a: None)
    assert "repeat" not in str(e.value)


def test_think_false_per_call_turns_thinking_off_and_sends_options(stub):
    stub.responses = [(200, {"message": {"content": "a"}, "done": True})]
    make_client(cfg_for(stub, top_p=0.9, top_k=30)).chat(MSGS, think=False)
    body = stub.requests[0][2]
    assert body["think"] is False and body["options"]["top_p"] == 0.9 and body["options"]["top_k"] == 30


def test_no_answer_is_still_an_llm_error():
    assert issubclass(NoAnswer, LLMError)


# --- the runner treats a looping/empty attempt as a failed attempt, not a pause --

def test_looping_attempt_is_retried_without_thinking_and_with_feedback(tmp_path, cfg):  # noqa: F811
    llm = FakeLLM(NoAnswer("the model got stuck repeating itself while thinking and was stopped"), BODY_OK)
    r, _ = make_body_runner(tmp_path, llm, cfg)
    assert r.run().outcome == "done"
    assert "think" not in llm.kwargs[0] or llm.kwargs[0]["think"] is not False
    assert llm.kwargs[1]["think"] is False
    assert "stuck repeating" in llm.calls[1][1]["content"]
    assert llm.kwargs[1]["temperature"] > llm.kwargs[0]["temperature"]


def test_a_real_server_error_still_pauses(tmp_path, cfg):  # noqa: F811
    r, _ = make_body_runner(tmp_path, FakeLLM(LLMError("cannot reach the model server")), cfg)
    assert r.run().outcome == "paused"


def test_every_call_is_streamed_so_a_loop_can_be_stopped(tmp_path, cfg):  # noqa: F811
    llm = FakeLLM(BODY_OK)
    make_body_runner(tmp_path, llm, cfg)[0].run()
    assert callable(llm.kwargs[0]["on_token"])
    json.dumps(llm.kwargs[0].get("stop"))
