"""Claude Haiku 5.5 (Anthropic API) in the model chain. No live calls: the SDK talks to an httpx2.MockTransport."""
import json
import sys
from pathlib import Path

import pytest

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import budget  # noqa: E402
import llm  # noqa: E402
from test_llm import FakeResp, sse  # noqa: E402

SYSTEM = "You are the website assistant for shop.example. Answer only from the pages."
MSGS = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "How long does delivery take?"}]
CLAUDE = llm.CLAUDE_MODEL


def claude_sse(*texts, input_tokens=1000, output_tokens=40, stop="end_turn", error_after=None, thinking=True):
    """An Anthropic Messages stream: optional thinking block, then the texts. error_after=n: an error event after n texts."""
    start = {"id": "msg_1", "type": "message", "role": "assistant", "model": CLAUDE, "content": [], "stop_reason": None,
             "stop_sequence": None, "usage": {"input_tokens": input_tokens, "output_tokens": 1}}
    ev = [("message_start", {"type": "message_start", "message": start})]
    i = 0
    if thinking:
        ev += [("content_block_start", {"type": "content_block_start", "index": 0,
                                        "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
               ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                        "delta": {"type": "signature_delta", "signature": "sig"}}),
               ("content_block_stop", {"type": "content_block_stop", "index": 0})]
        i = 1
    ev.append(("content_block_start", {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}}))
    for n, text in enumerate(texts):
        if n == error_after:
            ev.append(("error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}))
            break
        ev.append(("content_block_delta", {"type": "content_block_delta", "index": i,
                                           "delta": {"type": "text_delta", "text": text}}))
    else:
        ev += [("content_block_stop", {"type": "content_block_stop", "index": i}),
               ("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                                  "usage": {"output_tokens": output_tokens}}),
               ("message_stop", {"type": "message_stop"})]
    body = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in ev)
    return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())


def claude_json(text, input_tokens=500, output_tokens=30, stop="end_turn"):
    content = [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": text}]
    return httpx2.Response(200, json={"id": "msg_1", "type": "message", "role": "assistant", "model": CLAUDE,
                                      "content": content, "stop_reason": stop, "stop_sequence": None,
                                      "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}})


def claude_error(status, kind, message="error"):
    return httpx2.Response(status, json={"type": "error", "error": {"type": kind, "message": message}})


def cost(inp, out):
    return (inp * 0.10 + out * 0.50) / 1_000_000


@pytest.fixture
def claude(monkeypatch):
    """Turns Claude on with a fake API: append httpx2.Response objects (or exceptions) to `replies`; `seen` gets
    each request body."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    seen, replies = [], []

    def handler(request):
        seen.append(json.loads(request.content))
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    client = anthropic.Anthropic(api_key="sk-ant-test", max_retries=0,
                                 http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    monkeypatch.setattr(llm, "_claude_client", lambda: client)
    return seen, replies


@pytest.fixture
def openrouter(monkeypatch):
    """OpenRouter with two paid models and one free one; append FakeResp replies."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODELS", "p1,p2,f1:free")
    monkeypatch.setattr(llm, "_KEY_STATE", {"state": "unchecked", "key": None})
    calls, replies = [], []

    def post(url, headers=None, json=None, stream=False, timeout=None):
        calls.append(json)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(llm.requests, "post", post)
    return calls, replies


# ---- order and switches -------------------------------------------------------------------------------------

def test_without_key_the_chain_and_answers_are_unchanged(openrouter):
    calls, replies = openrouter
    assert llm.chain() == llm.models() == ["p1", "p2", "f1:free"] and not llm.claude_on()
    replies += [FakeResp(402), FakeResp(402), FakeResp(lines=sse("free answer"))]
    assert "".join(llm.stream_text(MSGS)) == "free answer"  # the autouse fixture fails the test if Claude is called
    assert [c["model"] for c in calls] == ["p1", "p2", "f1:free"]
    assert "claude" not in json.dumps(llm.chain())


def test_claude_comes_after_paid_models_and_before_free_ones(monkeypatch, claude):
    monkeypatch.setenv("OPENROUTER_MODELS", "p1,p2,f1:free,f2:free")
    assert llm.chain() == ["p1", "p2", CLAUDE, "f1:free", "f2:free"]
    monkeypatch.setenv("OPENROUTER_MODELS", "p1")
    assert llm.chain() == ["p1", CLAUDE]
    monkeypatch.delenv("OPENROUTER_MODELS")
    assert llm.chain()[:3] == llm.DEFAULT_MODELS[:2] + [CLAUDE] and llm.chain()[3].endswith(":free")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  ")
    assert CLAUDE not in llm.chain()


# ---- streaming ----------------------------------------------------------------------------------------------

def test_stream_uses_claude_when_openrouter_has_no_credit(openrouter, claude):
    calls, replies = openrouter
    seen, answers = claude
    replies += [FakeResp(402), FakeResp(402)]
    answers.append(claude_sse("Delivery takes ", "2 days [1].", input_tokens=1000, output_tokens=40))
    usages = []
    out = "".join(llm.stream_text(MSGS, max_tokens=1500, on_usage=usages.append))
    assert out == "Delivery takes 2 days [1]."  # text only: the thinking block is not shown
    assert [c["model"] for c in calls] == ["p1", "p2"]  # the free model was never asked
    body = seen[0]
    assert body["model"] == CLAUDE and body["max_tokens"] == 1500 and body["stream"] is True
    assert body["system"] == SYSTEM  # the same system prompt, word for word
    assert body["messages"] == [{"role": "user", "content": "How long does delivery take?"}]
    assert body["output_config"] == {"effort": "low"} and "thinking" not in body
    assert usages == [llm.Usage(1000, 40, pytest.approx(cost(1000, 40)), CLAUDE, 1)]
    snap = llm.claude_snapshot()
    assert snap["spent_usd"] == pytest.approx(cost(1000, 40), abs=1e-6) and snap["in_flight_usd"] == 0
    assert snap["cap_usd"] == 1.0 and snap["model"] == CLAUDE and snap["on"] is True


def test_claude_error_before_text_falls_back_to_the_free_model(openrouter, claude):
    calls, replies = openrouter
    _, answers = claude
    replies += [FakeResp(402), FakeResp(402), FakeResp(lines=sse("free"))]
    answers.append(claude_error(529, "overloaded_error"))
    assert "".join(llm.stream_text(MSGS)) == "free"
    assert [c["model"] for c in calls] == ["p1", "p2", "f1:free"]
    assert llm.claude_snapshot()["spent_usd"] == 0  # an error answer is not billed


def test_claude_error_mid_answer_ends_with_the_interrupted_line(openrouter, claude):
    calls, replies = openrouter
    _, answers = claude
    replies += [FakeResp(402), FakeResp(402)]
    answers.append(claude_sse("part one", "two", error_after=1))
    usages = []
    out = "".join(llm.stream_text(MSGS, on_usage=usages.append))
    assert out == "part one" + llm.INTERRUPTED and len(calls) == 2
    assert usages[0].model == CLAUDE and usages[0].calls == 1
    snap = llm.claude_snapshot()
    assert snap["in_flight_usd"] == 0 and snap["spent_usd"] > 0  # an unfinished call counts its worst case


def test_reader_leaving_early_still_counts_the_claude_call(openrouter, claude):
    _, replies = openrouter
    _, answers = claude
    replies += [FakeResp(402), FakeResp(402)]
    answers.append(claude_sse("first ", "second ", "third"))
    usages = []
    gen = llm.stream_text(MSGS, on_usage=usages.append)
    assert next(gen) == "first "
    gen.close()
    assert len(usages) == 1 and usages[0].model == CLAUDE
    snap = llm.claude_snapshot()
    assert snap["in_flight_usd"] == 0 and snap["spent_usd"] > 0


def test_refusal_before_text_moves_on_and_its_cost_is_counted(openrouter, claude):
    _, replies = openrouter
    _, answers = claude
    replies += [FakeResp(402), FakeResp(402), FakeResp(lines=sse("free"))]
    answers.append(claude_sse(stop="refusal", input_tokens=900, output_tokens=5))
    usages = []
    assert "".join(llm.stream_text(MSGS, on_usage=usages.append)) == "free"
    assert usages[0].model == "f1:free" and usages[0].calls == 2  # the billed Claude attempt is included
    assert usages[0].cost_usd == pytest.approx(cost(900, 5))


def test_daily_cap_skips_claude(monkeypatch, openrouter, claude):
    _, replies = openrouter
    seen, answers = claude
    monkeypatch.setenv("ANTHROPIC_DAILY_CAP", "0")
    replies += [FakeResp(402), FakeResp(402), FakeResp(lines=sse("free"))]
    assert "".join(llm.stream_text(MSGS)) == "free" and seen == []


def test_cap_stops_claude_once_today_is_spent(monkeypatch, openrouter, claude):
    _, replies = openrouter
    seen, answers = claude
    replies += [FakeResp(402), FakeResp(402)]
    answers.append(claude_sse("one", input_tokens=2000, output_tokens=100))
    assert "".join(llm.stream_text(MSGS)) == "one"
    spent = llm.claude_snapshot()["spent_usd"]
    monkeypatch.setenv("ANTHROPIC_DAILY_CAP", f"{spent + 0.0001:.6f}")  # less room than one more call may cost
    replies += [FakeResp(402), FakeResp(402), FakeResp(lines=sse("free"))]
    assert "".join(llm.stream_text(MSGS)) == "free" and len(seen) == 1
    for bad in ("abc", "nan", "inf", "-3"):
        monkeypatch.setenv("ANTHROPIC_DAILY_CAP", bad)
        assert llm.claude_cap().daily_usd == (0.0 if bad == "-3" else 1.0)


def test_rejected_claude_key_is_not_tried_again(openrouter, claude):
    _, replies = openrouter
    seen, answers = claude
    replies += [FakeResp(402), FakeResp(402), FakeResp(lines=sse("free"))]
    answers.append(claude_error(401, "authentication_error", "invalid x-api-key"))
    assert "".join(llm.stream_text(MSGS)) == "free"
    assert not llm.claude_on() and CLAUDE not in llm.chain()
    replies += [FakeResp(402), FakeResp(402), FakeResp(lines=sse("free again"))]
    assert "".join(llm.stream_text(MSGS)) == "free again" and len(seen) == 1


def test_claude_alone_answers_without_an_openrouter_key(monkeypatch, claude):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("openrouter_key", raising=False)
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: pytest.fail("OpenRouter called without a key"))
    _, answers = claude
    answers.append(claude_sse("only Claude"))
    assert llm.available() and "".join(llm.stream_text(MSGS)) == "only Claude"


def test_rejected_openrouter_key_hands_over_to_claude(openrouter, claude):
    calls, replies = openrouter
    _, answers = claude
    replies.append(FakeResp(401))
    answers.append(claude_sse("from Claude"))
    assert "".join(llm.stream_text(MSGS)) == "from Claude"
    assert llm.key_state() == "rejected" and len(calls) == 1 and llm.available()


# ---- JSON and the small extras ------------------------------------------------------------------------------

SCHEMA = {"type": "object", "properties": {"rows": {"type": "array", "items": {"type": "string"}}},
          "required": ["rows"], "additionalProperties": False}


def test_complete_json_uses_claude_structured_output(openrouter, claude):
    _, replies = openrouter
    seen, answers = claude
    replies += [FakeResp(402), FakeResp(402)]
    answers.append(claude_json('{"rows": ["a", "b"]}'))
    data, model = llm.complete_json([{"role": "user", "content": "Extract the rows"}], SCHEMA, "rows")
    assert data == {"rows": ["a", "b"]} and model == CLAUDE
    assert seen[0]["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}
    assert seen[0]["max_tokens"] == 4000 and "system" not in seen[0]
    assert llm.claude_snapshot()["spent_usd"] == pytest.approx(cost(500, 30), abs=1e-6)
    assert llm.estimate_cost(CLAUDE, 1_000_000, 0) == pytest.approx(0.10)  # main.py's extraction estimate


def test_complete_json_moves_on_after_a_bad_claude_answer(openrouter, claude):
    _, replies = openrouter
    _, answers = claude
    body = {"choices": [{"message": {"content": '{"rows": ["free"]}'}}]}
    replies += [FakeResp(402), FakeResp(402), FakeResp(text=json.dumps(body))]
    answers.append(claude_json("not json"))
    assert llm.complete_json([{"role": "user", "content": "x"}], SCHEMA) == ({"rows": ["free"]}, "f1:free")


def test_quick_json_and_translation_fall_back_to_claude_with_thinking_off(openrouter, claude):
    calls, replies = openrouter
    seen, answers = claude
    replies += [FakeResp(402), FakeResp(500)]
    answers += [claude_json('{"rows": ["q1"]}', 300, 20), claude_json("shipping cost to Morocco", 60, 6)]
    data, usage = llm.quick_json([{"role": "user", "content": "starters"}], SCHEMA)
    assert data == {"rows": ["q1"]} and usage.model == CLAUDE and usage.cost_usd == pytest.approx(cost(300, 20))
    assert seen[0]["thinking"] == {"type": "disabled"} and seen[0]["max_tokens"] == 250
    got = []
    assert llm.translate_query("كم ثمن الشحن إلى المغرب", "latin", on_usage=got.append) == "shipping cost to Morocco"
    assert seen[1]["thinking"] == {"type": "disabled"} and seen[1]["max_tokens"] == 80
    assert seen[1]["system"].startswith("Translate the user's search question into English")
    assert got[0].model == CLAUDE and [c["model"] for c in calls] == [llm.TRANSLATE_MODEL, llm.TRANSLATE_MODEL]


def test_extras_without_claude_key_fail_as_before(openrouter):
    _, replies = openrouter
    replies += [FakeResp(402), FakeResp(500)]
    with pytest.raises(llm.LLMError):
        llm.quick_json([{"role": "user", "content": "x"}], SCHEMA)
    assert llm.translate_query("bonjour", "latin") == "bonjour"


def test_message_conversion():
    msgs = [{"role": "system", "content": "A"}, {"role": "assistant", "content": "orphan"},
            {"role": "user", "content": "q1"}, {"role": "assistant", "content": " "}, {"role": "user", "content": "q2"}]
    system, turns = llm._claude_messages(msgs)
    assert system == "A" and turns == [{"role": "user", "content": "q1"}, {"role": "user", "content": "q2"}]
    assert llm._claude_cost(200_000, 0) == pytest.approx(0.10)  # prompts over 100,000 tokens: $0.50 / MTok


# ---- the cap itself ---------------------------------------------------------------------------------------

def test_daily_cap_holds_settles_and_resets_at_midnight():
    t = [1_800_000_000.0]
    cap = budget.DailyCap(0.01, clock=lambda: t[0])
    a = cap.hold(0.006)
    assert a is not None and cap.hold(0.005) is None  # the hold counts before the call ends
    cap.settle(a, 0.001)  # the real cost replaces the worst case
    b = cap.hold(0.005)
    assert b is not None and cap.snapshot()["spent_usd"] == 0.001
    t[0] += 86_400  # next day: fresh counters, and yesterday's ticket frees nothing
    cap.settle(b, 0.002)
    assert cap.snapshot()["spent_usd"] == 0.002 and cap.snapshot()["in_flight_usd"] == 0
    assert cap.hold(0.008) is not None and cap.hold(0.0001) is None


# ---- through the API ----------------------------------------------------------------------------------------

def test_chat_answers_with_claude_end_to_end(openrouter, claude):
    from fastapi.testclient import TestClient

    from api import main
    from test_api import ask
    _, replies = openrouter
    seen, answers = claude
    replies += [FakeResp(402), FakeResp(402)]
    answers.append(claude_sse("Delivery to Casablanca takes 2 days [1].\nFOLLOWUPS: Is it free?"))
    app = main.create_app(translate=lambda t, lang: t, check_key=False)  # the real llm.stream_text
    done = dict(ask(TestClient(app), "How long does delivery to Casablanca take?"))["done"]
    assert done["model"] == CLAUDE and done["status"] == "answered" and done["cost"] > 0 and done["cited"] == [1]
    assert seen[0]["system"].startswith("You are the website assistant")


def test_healthz_shows_claude_only_with_its_key(monkeypatch, claude):
    from fastapi.testclient import TestClient

    from api import main
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("openrouter_key", raising=False)
    client = TestClient(main.create_app(translate=lambda t, lang: t, check_key=False))
    h = client.get("/healthz").json()
    assert h["ai"] is True and h["ai_key"] == "missing"
    assert h["claude"] == {"model": CLAUDE, "on": True, "day": h["claude"]["day"], "spent_usd": 0.0, "cap_usd": 1.0,
                           "in_flight_usd": 0.0}
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    h = client.get("/healthz").json()
    assert h["ai"] is False and "claude" not in h
