"""Smoke tests for eval/run.py: no network, no API key."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import run as ev  # noqa: E402

ROWS = [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").read_text().splitlines() if line.strip()]


def test_question_set_shape():
    assert len(ROWS) == 26 and len({r["id"] for r in ROWS}) == 26
    assert sum(1 for r in ROWS if not r["prev"] and r["lang"] == "en") == 18
    assert sum(1 for r in ROWS if r["prev"]) == 5 and all(r["follows"] for r in ROWS if r["prev"])
    assert sum(1 for r in ROWS if r["lang"] != "en") == 3 and any(r["lang"] == "ar" for r in ROWS)
    assert sum(1 for r in ROWS if not r["answerable"]) >= 3
    for r in ROWS:  # every expected page exists in its frozen fixture
        urls = {p["url"] for p in json.loads((ROOT / "eval" / "fixtures" / f"{r['fixture']}.json").read_text())}
        assert set(r["expect_urls"]) <= urls, r["id"]


def test_retrieval_mode_runs_offline():
    report = ev.evaluate(ROOT, "retrieval", ROWS)
    assert len(report["items"]) == 26
    assert report["summary"]["hit4"].startswith(tuple("0123456789"))
    assert all(i["passages"] <= 6 for i in report["items"])


def test_full_mode_with_fake_stream_scores_answers():
    rows = [r for r in ROWS if r["id"] in ("shop-returns", "shop-worldcup")]

    def fake_stream(messages, **kwargs):
        assert messages[0]["role"] == "system"
        yield "You have 14 days to return an unused item [1]."

    report = ev.evaluate(ROOT, "full", rows, stream=fake_stream)
    by_id = {i["id"]: i for i in report["items"]}
    assert by_id["shop-returns"]["correct"] is True
    assert by_id["shop-worldcup"]["correct"] is (by_id["shop-worldcup"]["status"] == "no_match")
    assert report["summary"]["errors"] == 0


def test_score_rejects_refusals_and_accepts_honest_no_answer():
    answerable = {"answerable": True, "expect_any": ["try"]}
    assert ev.score(answerable, "Use try and except [1].", "answered") == (True, False)
    assert ev.score(answerable, "I couldn't find that; see try/except docs.", "answered") == (False, True)
    assert ev.score(answerable, "لم أتمكن من العثور على إجابة. try", "answered")[0] is False
    assert ev.score({"answerable": False, "expect_any": []}, "x", "no_match") == (True, True)


def test_usage_spy_requests_and_reads_stream_usage():
    class FakeResp:
        status_code = 200

        def iter_lines(self, decode_unicode=False):
            yield 'data: {"choices":[{"delta":{"content":"hi"}}]}'
            yield 'data: {"choices":[],"usage":{"prompt_tokens":120,"completion_tokens":7,"cost":0.0004}}'
            yield "data: [DONE]"

    class FakeRequests:
        RequestException = Exception
        sent = None

        def post(self, url, json=None, **kwargs):
            FakeRequests.sent = json
            return FakeResp()

    spy = ev.UsageSpy(FakeRequests())
    resp = spy.post("u", json={"stream": True, "messages": [{"role": "user", "content": "abcd"}]})
    assert list(resp.iter_lines(decode_unicode=True))[-1] == "data: [DONE]"
    assert FakeRequests.sent["usage"] == {"include": True}
    assert spy.usage["prompt_tokens"] == 120 and spy.posts == 1 and spy.prompt_chars == 4
    assert spy.RequestException is Exception
    fields = ev._usage_fields(None, spy, 0, "")
    assert (fields["input_tokens"], fields["output_tokens"], fields["usage_source"]) == (120, 7, "api")
