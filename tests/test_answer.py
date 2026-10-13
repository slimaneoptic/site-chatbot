import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import llm  # noqa: E402
from answer import (  # noqa: E402
    ANSWER_MAX_TOKENS, NOT_FOUND_TEXT, ask, parse_citations, split_followups, trim_history,
)
from budget import Budget  # noqa: E402
from cache import AnswerCache  # noqa: E402
from search import Passage  # noqa: E402

PASSAGES = [Passage(f"https://shop.example/p{i}", f"Page {i}", f"Text of page {i}. " * 60) for i in range(1, 7)]
KEEP = lambda p, q: p  # noqa: E731 - no trimming, so tests do not depend on search.trim_passage


def found(matched=True, passages=PASSAGES):
    return SimpleNamespace(matched=matched, passages=list(passages) if matched else [], search_query="returns policy")


class FakeLLM:
    """Records every call; streams the given chunks and reports usage like llm.stream_text."""

    def __init__(self, chunks=("Returns are free [1].", "\nFOLLOWUPS: How long? | Who pays? | Where?"), error=None):
        self.chunks, self.error, self.calls = list(chunks), error, []

    def __call__(self, messages, *, max_tokens=4000, on_usage=None):
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        return self._gen(on_usage)

    def _gen(self, on_usage):
        for c in self.chunks:
            yield c
        if self.error:
            raise self.error
        if on_usage:
            on_usage(llm.Usage(prompt_tokens=900, completion_tokens=60, cost_usd=0.001, model="fake", calls=1))


def run(question="Can I return it?", history=(), **kw):
    kw.setdefault("trim", KEEP)
    r = ask("shop.example", question, kw.pop("found", found()), list(history), source="src1", session="s1", ip="1.2.3.4", **kw)
    chunks = list(r)
    return r, chunks


def test_no_match_makes_no_model_call_and_is_not_counted():
    fake, budget, cache = FakeLLM(), Budget(), AnswerCache()
    r, chunks = run(found=found(matched=False), stream=fake, budget=budget, cache=cache)
    assert chunks == [NOT_FOUND_TEXT] and r.result.status == "no_match"
    assert fake.calls == [] and r.result.usage.calls == 0
    assert budget.snapshot()["questions"] == 0 and len(cache) == 0


def test_answer_streams_without_followups_line_and_returns_usage():
    fake, budget = FakeLLM(["Returns are free", " within 30 days [1][2].\nFOLL", "OWUPS: How long? | Who pays [1]? | Where?"]), Budget()
    r, chunks = run(stream=fake, budget=budget)
    shown = "".join(chunks)
    assert "FOLL" not in shown and shown.rstrip() == "Returns are free within 30 days [1][2]."
    res = r.result
    assert res.status == "answered" and res.text == shown.rstrip()
    assert res.followups == ["How long?", "Who pays?", "Where?"]
    assert res.cited == [1, 2] and len(res.passages) == 4
    assert res.usage.calls == 1 and res.usage.prompt_tokens == 900
    assert fake.calls[0]["max_tokens"] == ANSWER_MAX_TOKENS
    user_msg = fake.calls[0]["messages"][-1]["content"]
    assert "[4] Page 4" in user_msg and "[5]" not in user_msg and "FOLLOWUPS" in fake.calls[0]["messages"][0]["content"]
    snap = budget.snapshot()
    assert snap["questions"] == 1 and snap["spent_usd"] == pytest.approx(0.001)


def test_marker_without_newline_and_text_with_f_words_are_handled():
    fake = FakeLLM(["Free shipping for", " orders of fifty f", "ollars. FOLLOWUPS: a? | b?"])
    r, chunks = run(stream=fake)
    assert "".join(chunks).strip() == "Free shipping for orders of fifty follars."
    assert r.result.followups == ["a?", "b?"]


def test_cache_hit_on_second_ask_and_no_cache_with_history():
    fake, cache = FakeLLM(), AnswerCache()
    run(stream=fake, cache=cache)
    r, chunks = run("can I return it", stream=fake, cache=cache)
    assert len(fake.calls) == 1 and r.result.status == "cached" and r.result.usage.calls == 0
    assert chunks == ["Returns are free [1]."] and r.result.followups == ["How long?", "Who pays?", "Where?"]
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    run(history=history, stream=fake, cache=cache)
    assert len(fake.calls) == 2


def test_budget_denial_gives_extractive_answer_with_message():
    fake, budget = FakeLLM(), Budget(session_cap=1)
    run(stream=fake, budget=budget)
    r, chunks = run("another question", stream=fake, budget=budget)
    assert len(fake.calls) == 1
    assert r.result.status == "extractive" and "https://shop.example/p1" in chunks[0]
    assert r.result.note and "reload" not in r.result.note.lower()


def test_llm_error_falls_back_to_extractive_and_is_not_cached():
    fake, cache, budget = FakeLLM(chunks=[], error=llm.LLMError("busy")), AnswerCache(), Budget()
    r, chunks = run(stream=fake, cache=cache, budget=budget)
    assert r.result.status == "error" and r.result.note == "AI busy"
    assert "Page 1" in chunks[0] and len(cache) == 0 and budget.snapshot()["questions"] == 0


def test_error_after_text_keeps_partial_answer():
    fake = FakeLLM(chunks=["Returns are"], error=llm.LLMError("cut"))
    r, chunks = run(stream=fake, cache=AnswerCache())
    assert "".join(chunks).startswith("Returns are") and "interrupted" in r.result.text
    assert r.result.status == "error"


def test_old_style_stream_and_usage_estimate():
    seen = []

    def old_stream(messages):
        seen.append(messages)
        yield "Answer [1]."

    r, chunks = run(stream=old_stream, budget=Budget())
    assert "".join(chunks) == "Answer [1]." and r.result.usage.calls == 1 and r.result.usage.prompt_tokens > 0


def test_default_trim_cuts_long_passages():
    text = "We ship parcels with a tracked carrier every weekday. " * 20 + "Our returns policy gives you 30 days. " + "Gift cards never expire. " * 20
    long = [Passage("https://shop.example/help", "Help", text)]
    r = ask("shop.example", "Can I return it?", found(passages=long), [], source="s", session="s", ip="i", stream=FakeLLM())
    list(r)
    assert len(r.result.passages[0].text) <= 700 and "returns policy" in r.result.passages[0].text


def test_trim_history_keeps_two_short_turns():
    long = "Returns take 30 days [1]. " * 40 + "\n\n**Sources**\n1. https://shop.example/p1"
    history = []
    for i in range(4):
        history += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": long, "sources": []}]
    out = trim_history(history)
    assert [m["content"] for m in out if m["role"] == "user"] == ["q2", "q3"]
    for m in out:
        assert set(m) == {"role", "content"}
        assert len(m["content"]) <= 301 and "[1]" not in m["content"] and "Sources" not in m["content"]
    assert trim_history([]) == []


def test_split_followups_and_citations():
    assert split_followups("Hi.\n**FOLLOWUPS:** a | b | c | d") == ("Hi.", ["a", "b", "c"])
    assert split_followups("No list here.") == ("No list here.", [])
    assert split_followups("Hi.\nFollow-ups: x?") == ("Hi.", ["x?"])
    assert parse_citations("a [1], b [2, 3] c [1][4] d [7] e [x]", 4) == [1, 2, 3, 4]
