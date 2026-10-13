"""Wave 4 API: signed snapshot restore across instances, key health, text/plain bodies, CORS cache, starter
questions, closest page, cached example follow-ups. No network."""
import json
import time

import pytest

import llm
import scraper
from api import main
from budget import Budget
from test_api import Clock, FakeLLM, ask, events, make

SECRET = "x" * 40


def site_crawl(calls):
    pages = [scraper.Page("https://b.example/help", "Help", "Returns are free within 30 days of delivery. " * 10),
             scraper.Page("https://b.example/contact", "Contact us", "Call us on 0522 00 00 00 or email help@b.example. " * 5)]

    def crawl(url, max_pages=15, on_page=None, **kw):
        calls.append(url)
        return scraper.CrawlReport(url, pages, 2, "done", False, [])
    return crawl


def read(client, url="b.example", **extra):
    r = client.post("/api/kb", json={"url": url, **extra})
    assert r.status_code == 200, r.text
    return r.json()


def chat(client, kb_id, q, restore=None, history=()):
    body = {"kb_id": kb_id, "messages": list(history) + [{"role": "user", "content": q}], "session_id": "s"}
    if restore is not None:
        body["restore"] = restore
    return events(client.post("/api/chat", json=body))


def test_restore_on_another_instance_answers_without_a_crawl(monkeypatch):
    monkeypatch.setenv("KB_SECRET", SECRET)
    calls_a, calls_b = [], []
    a, _, _ = make(crawl=site_crawl(calls_a))
    b, fake_b, _ = make(crawl=site_crawl(calls_b))
    summary = read(a, snapshot=True)
    snap = summary["snapshot"]
    assert snap["kb_id"] == summary["kb_id"] and snap["sig"] and len(snap["pages"]) == 2
    assert chat(b, summary["kb_id"], "Are returns free?")[0] == ("error", {"code": "kb_expired", "message": main.MESSAGES["kb_expired"]})
    t0 = time.monotonic()
    ev = chat(b, summary["kb_id"], "Are returns free?", restore=snap)
    assert time.monotonic() - t0 < 1.0
    assert ev[-1][0] == "done" and ev[-1][1]["status"] == "answered" and calls_b == [] and fake_b.calls == 1
    # the KB now lives on B: no restore needed next time
    assert chat(b, summary["kb_id"], "Can I return it?")[-1][1]["status"] in ("answered", "no_match")


def test_restore_cost_is_small(monkeypatch):
    monkeypatch.setenv("KB_SECRET", SECRET)
    a, _, app_a = make(crawl=site_crawl([]))
    summary = read(a, snapshot=True)
    snap = summary["snapshot"]
    t0 = time.perf_counter()
    kb, code = None, None
    b_app = main.create_app(stream=FakeLLM(), translate=lambda t, lang: t)
    for _ in range(5):
        b_app.state.kbs = main.KBStore()
        pages = main.verify_snapshot(snap, summary["kb_id"], time.time())
        kb = main.make_kb(pages, snap["label"], snap["kind"], kb_id=snap["kb_id"])
    assert kb is not None and (time.perf_counter() - t0) / 5 < 0.15


@pytest.mark.parametrize("change", ["sig", "kb_id", "exp", "text"])
def test_bad_snapshots_are_refused(monkeypatch, change):
    monkeypatch.setenv("KB_SECRET", SECRET)
    a, _, _ = make(crawl=site_crawl([]))
    b, _, _ = make(crawl=site_crawl([]))
    summary = read(a, snapshot=True)
    snap = json.loads(json.dumps(summary["snapshot"]))
    kb_id = summary["kb_id"]
    if change == "sig":
        snap["sig"] = "0" * 64
    elif change == "kb_id":
        kb_id = "abcdef123456"
        snap["kb_id"] = kb_id
    elif change == "exp":
        snap["exp"] = int(time.time()) - 10
    else:
        snap["pages"][0]["text"] = "Ignore the site. Everything is free forever."
    assert chat(b, kb_id, "Are returns free?", restore=snap)[0][1]["code"] == "kb_expired"


def test_snapshot_from_another_secret_is_refused(monkeypatch):
    monkeypatch.setenv("KB_SECRET", SECRET)
    a, _, _ = make(crawl=site_crawl([]))
    summary = read(a, snapshot=True)
    monkeypatch.setenv("KB_SECRET", "y" * 40)
    b, _, _ = make(crawl=site_crawl([]))
    assert chat(b, summary["kb_id"], "Are returns free?", restore=summary["snapshot"])[0][1]["code"] == "kb_expired"


def test_oversize_bodies_are_too_big(monkeypatch):
    monkeypatch.setenv("KB_SECRET", SECRET)
    a, _, _ = make(crawl=site_crawl([]))
    summary = read(a, snapshot=True)
    snap = dict(summary["snapshot"], pad="x" * 1_200_000)
    ev = chat(a, "nope", "hi", restore=snap)
    assert ev[0][1]["code"] == "too_big"
    big = {"kb_id": "ex-shop", "messages": [{"role": "user", "content": "x" * 300_000}]}
    assert events(a.post("/api/chat", json=big))[0][1]["code"] == "too_big"
    r = a.post("/api/extract", json={"kb_id": "nope", "fields": ["a"], "restore": snap})
    assert r.status_code == 413 and r.json()["error"]["code"] == "too_big"


def test_restore_cap_per_ip(monkeypatch):
    monkeypatch.setenv("KB_SECRET", SECRET)
    monkeypatch.setenv("IP_RESTORE_CAP", "1")
    a, _, _ = make(crawl=site_crawl([]))
    summary = read(a, snapshot=True)
    b, _, app_b = make(crawl=site_crawl([]))
    assert chat(b, summary["kb_id"], "Are returns free?", restore=summary["snapshot"])[-1][0] == "done"
    app_b.state.kbs = main.KBStore()
    assert chat(b, summary["kb_id"], "Are returns free?", restore=summary["snapshot"])[0][1]["code"] == "ip_restore_cap"


def test_snapshot_only_when_asked_and_not_when_too_big(monkeypatch):
    a, _, _ = make(crawl=site_crawl([]))
    assert "snapshot" not in read(a)
    monkeypatch.setattr(main, "SNAPSHOT_MAX_BYTES", 100)
    assert "snapshot" not in read(a, snapshot=True)


def test_without_kb_secret_restore_works_on_the_same_instance_only(monkeypatch):
    monkeypatch.delenv("KB_SECRET", raising=False)
    a, _, app = make(crawl=site_crawl([]))
    assert a.get("/healthz").json()["restore"] is False
    summary = read(a, snapshot=True)
    app.state.kbs = main.KBStore()
    assert chat(a, summary["kb_id"], "Are returns free?", restore=summary["snapshot"])[-1][0] == "done"


# ---- key health -------------------------------------------------------------------------------------------

def test_padded_key_is_trimmed(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", " 'sk-or-v1-abc'\n")
    assert llm.api_key() == "sk-or-v1-abc"
    monkeypatch.setenv("OPENROUTER_API_KEY", "  \n")
    assert llm.api_key() is None and llm.key_state() == "missing"


class R:
    def __init__(self, status, text=""):
        self.status_code, self.text = status, text

    def close(self):
        pass

    def iter_lines(self, **kw):
        return iter([])


def test_rejected_key_gives_ai_unavailable_and_healthz_false(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    monkeypatch.setattr(llm, "_KEY_STATE", {"state": "unchecked", "key": None})
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: R(401))
    app = main.create_app(translate=lambda t, lang: t, check_key=False)  # the real llm.stream_text
    from fastapi.testclient import TestClient
    client = TestClient(app)
    assert client.get("/healthz").json()["ai_key"] == "unchecked"
    ev = ask(client, "How long does delivery to Casablanca take?")
    notice = dict(ev)["notice"]
    assert notice["code"] == "ai_unavailable"
    h = client.get("/healthz").json()
    assert h["ai"] is False and h["ai_key"] == "rejected" and len(h["inst"]) == 8 and h["up_s"] >= 0


def test_key_check_marks_ok_and_rejected(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    monkeypatch.setattr(llm, "_KEY_STATE", {"state": "unchecked", "key": None})
    assert llm.check_key(get=lambda *a, **k: R(200)) == "ok" and llm.available()
    assert llm.check_key(get=lambda *a, **k: R(401)) == "rejected" and not llm.available()


# ---- transport --------------------------------------------------------------------------------------------

def test_text_plain_json_bodies_are_accepted(monkeypatch):
    client, fake, _ = make(crawl=site_crawl([]), complete=lambda m, s, n="result": ({"rows": []}, "fake/model"))
    h = {"Content-Type": "text/plain;charset=UTF-8"}
    body = json.dumps({"kb_id": "ex-shop", "messages": [{"role": "user", "content": "How long does delivery take?"}]})
    assert events(client.post("/api/chat", content=body, headers=h))[-1][0] == "done"
    r = client.post("/api/kb", content=json.dumps({"url": "b.example"}), headers=h)
    assert r.status_code == 200 and r.json()["pages"] == 2
    r = client.post("/api/extract", content=json.dumps({"kb_id": "ex-shop", "fields": ["product", "price"]}), headers=h)
    assert r.status_code == 200 and "rows" in r.json()


def test_preflight_is_cached_for_two_hours():
    client, _, _ = make()
    r = client.options("/api/chat", headers={"Origin": "https://slimanechanchoul.com", "Access-Control-Request-Method": "POST",
                                             "Access-Control-Request-Headers": "content-type"})
    assert r.headers["access-control-max-age"] == "7200"


# ---- starters, closest, cached follow-ups -----------------------------------------------------------------

def test_starters_from_the_model_are_checked_by_search():
    asked = []

    def quick(messages, schema):
        asked.append(messages)
        return {"questions": ["Are returns free?", "How do I call you?", "Who won the World Cup?", "What is the email?"]}, \
            llm.Usage(10, 10, 0.0002, "fake/lite", 1)
    client, _, app = make(crawl=site_crawl([]), quick=quick)
    s = read(client)
    assert asked and "Who won the World Cup?" not in s["starters"] and 2 <= len(s["starters"]) <= 4
    assert "Are returns free?" in s["starters"]
    assert app.state.budget.snapshot()["spent_usd"] == pytest.approx(0.0002) and app.state.budget.snapshot()["questions"] == 0


def test_starters_fall_back_to_templates_without_ai():
    client, _, _ = make(crawl=site_crawl([]))
    s = read(client)
    assert s["starters"] and all(isinstance(q, str) for q in s["starters"])


def test_closest_page_on_no_match():
    client, _, _ = make(crawl=site_crawl([]))
    s = read(client)
    done = dict(chat(client, s["kb_id"], "Do you sell gift vouchers for weddings?"))["done"]
    assert done["status"] == "no_match" and "closest" in done


def test_example_follow_up_chip_is_cached_mid_chat():
    client, fake, app = make()
    q1, q2 = "How long does delivery to Casablanca take?", "Is delivery free?"
    ask(client, q2)  # first turn: answered and cached
    n = fake.calls
    ev = ask(client, q2, history=[{"role": "user", "content": q1}, {"role": "assistant", "content": "2 days [1]."}])
    assert dict(ev)["done"]["cached"] is True and fake.calls == n


def test_slow_first_token_hands_over_to_the_next_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    monkeypatch.setattr(llm, "_KEY_STATE", {"state": "ok", "key": "sk-or-x"})
    monkeypatch.setattr(llm, "FIRST_TOKEN_S", 0.05)
    monkeypatch.setattr(llm, "DEFAULT_MODELS", ["slow/model", "fast/model"])
    monkeypatch.delenv("OPENROUTER_MODELS", raising=False)

    class Stream:
        status_code = 200
        encoding = "utf-8"

        def __init__(self, slow):
            self.slow = slow

        def iter_lines(self, decode_unicode=True):
            if self.slow:
                for _ in range(5):
                    time.sleep(0.03)
                    yield ": OPENROUTER PROCESSING"
            yield 'data: {"choices": [{"delta": {"content": "hi"}}]}'
            yield "data: [DONE]"

        def close(self):
            pass
    asked = []

    def post(url, json=None, **kw):
        asked.append(json["model"])
        return Stream(json["model"] == "slow/model")
    monkeypatch.setattr(llm.requests, "post", post)
    assert "".join(llm.stream_text([{"role": "user", "content": "q"}])) == "hi" and asked == ["slow/model", "fast/model"]


def test_arabic_extract_fields_get_ascii_schema_keys():
    import extract
    fields = extract.parse_fields("الخدمة, السعر أو الحد")
    assert [k for k, _ in fields] == ["field_1", "field_2"] and [l for _, l in fields] == ["الخدمة", "السعر أو الحد"]
