import json
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import llm  # noqa: E402

MSGS = [{"role": "user", "content": "x" * 400}]


class FakeResp:
    def __init__(self, status=200, lines=(), boom_after=None, text=""):
        self.status_code, self.lines, self.boom_after, self.text, self.encoding = status, list(lines), boom_after, text, None

    def iter_lines(self, decode_unicode=True):
        for i, line in enumerate(self.lines):
            if self.boom_after is not None and i == self.boom_after:
                raise requests.exceptions.ChunkedEncodingError("connection dropped")
            yield line

    def close(self):
        pass


def sse(*texts, usage=None):
    out = [f"data: {json.dumps({'choices': [{'delta': {'content': t}}]})}" for t in texts]
    if usage:
        out.append(f"data: {json.dumps({'choices': [], 'usage': usage})}")
    return out + ["data: [DONE]"]


@pytest.fixture
def fake_post(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODELS", "m1,m2")
    calls, replies = [], []

    def post(url, headers=None, json=None, stream=False, timeout=None):
        calls.append(json)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(llm.requests, "post", post)
    return calls, replies


def test_403_falls_back_and_usage_is_reported_once(fake_post):
    calls, replies = fake_post
    replies += [FakeResp(403), FakeResp(lines=sse("Hel", "lo", usage={"prompt_tokens": 50, "completion_tokens": 7, "cost": 0.0002}))]
    seen = []
    out = "".join(llm.stream_text(MSGS, max_tokens=1500, on_usage=seen.append, session_id="abc"))
    assert out == "Hello" and [c["model"] for c in calls] == ["m1", "m2"]
    assert calls[1]["max_tokens"] == 1500 and calls[1]["user"] == "abc"
    assert seen == [llm.Usage(50, 7, 0.0002, "m2", 1)]
    assert 403 in llm.RETRYABLE


def test_network_error_before_text_tries_next_model(fake_post):
    _, replies = fake_post
    replies += [FakeResp(lines=sse("a"), boom_after=0), FakeResp(lines=sse("ok"))]
    seen = []
    assert "".join(llm.stream_text(MSGS, on_usage=seen.append)) == "ok"
    assert seen[0].model == "m2" and seen[0].prompt_tokens == 100  # chars/4 estimate without API usage


def test_network_error_after_text_ends_with_interrupted_line(fake_post):
    _, replies = fake_post
    replies += [FakeResp(lines=sse("part one", "two"), boom_after=1)]
    out = "".join(llm.stream_text(MSGS))
    assert out.startswith("part one") and "interrupted" in out


def test_all_fail_and_deadline_raise_llm_error(fake_post, monkeypatch):
    _, replies = fake_post
    replies += [requests.exceptions.ConnectTimeout(), FakeResp(503)]
    with pytest.raises(llm.LLMError):
        list(llm.stream_text(MSGS))
    with pytest.raises(llm.LLMError):
        list(llm.stream_text(MSGS, deadline_s=0))


def test_cost_estimate_from_price_table():
    assert llm.estimate_cost("google/gemini-3.8-flash", 1_000_000, 0) == pytest.approx(0.75)
    assert llm.estimate_cost("unknown/free", 1000, 1000) == 0


def test_translate_query_success_and_failures(fake_post):
    calls, replies = fake_post
    body = {"choices": [{"message": {"content": "shipping cost to Morocco"}}], "usage": {"prompt_tokens": 20, "completion_tokens": 5}}
    replies += [FakeResp(text=json.dumps(body)), FakeResp(500), requests.exceptions.ReadTimeout()]
    seen = []
    assert llm.translate_query("كم ثمن الشحن إلى المغرب", "latin", on_usage=seen.append) == "shipping cost to Morocco"
    assert calls[0]["max_tokens"] <= 80 and calls[0]["model"] == llm.TRANSLATE_MODEL and seen[0].calls == 1
    assert llm.translate_query("bonjour", "latin") == "bonjour"
    assert llm.translate_query("bonjour", "latin") == "bonjour"


def test_translate_query_without_key_returns_text(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("openrouter_key", raising=False)
    assert llm.translate_query("hola", "latin") == "hola"
