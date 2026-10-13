import io
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient  # noqa: E402

import llm  # noqa: E402
import scraper  # noqa: E402
from api import main  # noqa: E402
from budget import Budget  # noqa: E402


class FakeLLM:
    def __init__(self, text="Delivery to Casablanca takes 2 days [1].\nFOLLOWUPS: Is it free? | Can I track it?"):
        self.calls = 0
        self.text = text

    def __call__(self, messages, max_tokens=4000, on_usage=None, **kw):
        self.calls += 1
        for i in range(0, len(self.text), 7):
            yield self.text[i:i + 7]
        if on_usage:
            on_usage(llm.Usage(100, 20, 0.0001, "fake/model", 1))


class Clock:
    def __init__(self):
        self.t = 1_800_000_000.0

    def __call__(self):
        return self.t


def events(resp) -> list[tuple[str, object]]:
    out = []
    for block in resp.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def make(**kw):
    fake = kw.pop("stream", None) or FakeLLM()
    app = main.create_app(stream=fake, translate=lambda t, lang: t, **kw)
    return TestClient(app), fake, app


def ask(client, q, kb="ex-shop", session="s1", history=()):
    msgs = list(history) + [{"role": "user", "content": q}]
    return events(client.post("/api/chat", json={"kb_id": kb, "messages": msgs, "session_id": session}))


def test_healthz_and_examples():
    client, _, _ = make()
    assert client.get("/healthz").json()["ok"] is True
    ex = client.get("/api/examples").json()["examples"]
    ids = {e["kb_id"] for e in ex}
    assert {"ex-shop", "ex-shop_ar", "ex-python_tutorial"} <= ids
    shop = next(e for e in ex if e["kb_id"] == "ex-shop")
    assert shop["pages"] > 0 and shop["words"] > 0 and shop["starters"]
    assert all(s["cached"] is False for s in shop["starters"])


def test_chat_event_order_and_payloads():
    client, fake, _ = make()
    ev = ask(client, "How long does delivery to Casablanca take?")
    names = [e for e, _ in ev]
    assert names[:2] == ["meta", "sources"]
    assert names[-2:] == ["followups", "done"]
    assert set(names[2:-2]) == {"delta"} and len(names[2:-2]) > 1
    meta, sources = ev[0][1], ev[1][1]
    assert meta["lang"] == "latin" and meta["dir"] == "ltr" and meta["pages"] > 0
    assert sources and sources[0]["n"] == 1 and sources[0]["url"].startswith("https://")
    assert set(sources[0]) >= {"n", "title", "url", "passage", "hl"}
    assert "<" not in sources[0]["passage"] or "</" not in sources[0]["passage"]
    text = "".join(d["text"] for e, d in ev if e == "delta")
    assert "FOLLOWUPS" not in text and "2 days" in text
    assert ev[-2][1] == ["Is it free?", "Can I track it?"]
    done = ev[-1][1]
    assert done["cached"] is False and done["status"] == "answered" and done["cost"] == pytest.approx(0.0001)
    assert done["left_today"] == 11 and isinstance(done["ttfw_ms"], int) and done["cited"] == [1]
    assert fake.calls == 1


def test_second_ask_is_cached_and_listed_in_examples():
    client, fake, _ = make()
    q = "How long does delivery to Casablanca take?"
    ask(client, q)
    ev = ask(client, q, session="s2")
    assert ev[-1][1]["cached"] is True and ev[-1][1]["cost"] == 0
    assert fake.calls == 1
    shop = next(e for e in client.get("/api/examples").json()["examples"] if e["kb_id"] == "ex-shop")
    starter = next(s for s in shop["starters"] if s["q"] == q)
    assert starter["cached"] is True and "2 days" in starter["answer"]["text"]


def test_no_match_makes_no_model_call():
    client, fake, _ = make()
    ev = ask(client, "Who won the 2022 World Cup?")
    assert [e for e, _ in ev] == ["meta", "sources", "delta", "followups", "done"]
    assert ev[1][1] == []
    assert ev[-1][1]["status"] == "no_match" and ev[-1][1]["cost"] == 0
    assert fake.calls == 0


def test_session_cap_falls_back_to_passages_with_notice():
    client, fake, _ = make(budget=Budget(session_cap=1, ip_cap=30))
    ask(client, "How long does delivery to Casablanca take?", session="a")
    ev = ask(client, "Can I pay cash on delivery?", session="a")
    names = [e for e, _ in ev]
    assert names[:3] == ["meta", "sources", "notice"] and names[-1] == "done"
    assert ev[2][1]["code"] == "session_cap" and "reload" not in ev[2][1]["message"].lower()
    assert ev[-1][1]["status"] == "extractive" and ev[-1][1]["left_today"] == 0
    assert fake.calls == 1


def test_global_budget_falls_back():
    client, fake, _ = make(budget=Budget(daily_usd=0.0))
    ev = ask(client, "Can I pay cash on delivery?")
    assert ("notice", ev[2][1]) == ev[2] and ev[2][1]["code"] == "global_budget"
    assert fake.calls == 0


def test_kb_expired_after_ttl():
    clock = Clock()
    client, _, app = make(clock=clock)
    kb = main.make_kb([scraper.Page("https://a.example/", "A", "Opening hours are nine to five every weekday. " * 10)],
                      "a.example", "site")
    app.state.kbs.put(kb)
    assert ask(client, "What are the opening hours?", kb=kb.id)[0][0] == "meta"
    clock.t += 3601
    ev = ask(client, "What are the opening hours?", kb=kb.id)
    assert ev == [("error", {"code": "kb_expired", "message": main.MESSAGES["kb_expired"]})]
    assert ask(client, "hi", kb="ex-shop")[0][0] == "meta"  # examples never expire


def test_bad_request():
    client, _, _ = make()
    ev = events(client.post("/api/chat", json={"kb_id": "ex-shop", "messages": []}))
    assert ev[0][0] == "error" and ev[0][1]["code"] == "bad_request"


def test_cors_allowlist():
    client, _, _ = make()
    for origin, ok in [("https://slimanechanchoul.com", True), ("http://localhost:5173", True),
                       ("http://127.0.0.1:8080", True), ("https://evil.example", False),
                       ("http://localhost.evil.example", False)]:
        r = client.options("/api/chat", headers={"Origin": origin, "Access-Control-Request-Method": "POST"})
        assert (r.headers.get("access-control-allow-origin") == origin) is ok, origin


def test_kb_from_url_json_and_stream():
    pages = [scraper.Page("https://b.example/help", "Help", "Returns are free within 30 days of delivery. " * 10)]

    def crawl(url, max_pages=15, on_page=None, **kw):
        if "127.0.0.1" in url:
            scraper.check_public_url(url)  # what the real read_site does first (no DNS for an IP literal)
        if on_page:
            on_page(1, pages[0].url)
        return scraper.CrawlReport(url, pages, 1, "done", False, [])

    client, _, _ = make(crawl=crawl)
    r = client.post("/api/kb", json={"url": "b.example/help"}).json()
    assert r["pages"] == 1 and r["words"] > 0 and r["label"] == "b.example/help" and r["warnings"] == []
    ev = events(client.post("/api/kb?stream=1", json={"url": "b.example/help"}))
    assert [e for e, _ in ev] == ["start", "progress", "done"] and ev[-1][1]["pages"] == 1
    ev = ask(client, "Are returns free?", kb=ev[-1][1]["kb_id"])
    assert ev[0][0] == "meta"
    bad = client.post("/api/kb", json={"url": "http://127.0.0.1/"})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "blocked"
    assert client.post("/api/kb", json={"url": "not a url"}).json()["error"]["code"] == "bad_url"


def test_kb_crawl_cap():
    calls = []

    def crawl(url, max_pages=15, on_page=None, **kw):
        calls.append(url)
        return scraper.CrawlReport(url, [scraper.Page(url, "T", "word " * 50)], 1, "done", False, [])

    client, _, _ = make(crawl=crawl, budget=Budget(ip_crawl_cap=1))
    assert client.post("/api/kb", json={"url": "c.example"}).status_code == 200
    r = client.post("/api/kb", json={"url": "c.example"})
    assert r.status_code == 429 and r.json()["error"]["code"] == "ip_crawl_cap" and len(calls) == 1


def test_pdf_upload_and_exports():
    pypdf = pytest.importorskip("pypdf")
    w = pypdf.PdfWriter()
    w.add_blank_page(200, 200)
    buf = io.BytesIO()
    w.write(buf)
    client, _, _ = make()
    r = client.post("/api/kb", files={"files": ("blank.pdf", buf.getvalue(), "application/pdf")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "no_text"


def test_extract_rows_with_source_and_csv():
    def complete(messages, schema, name="result"):
        return {"rows": [{"method": "Cash on delivery"}, {"method": "=HYPERLINK(1)"}]}, "fake/model"

    client, _, app = make(complete=complete)
    r = client.post("/api/extract", json={"kb_id": "ex-shop", "fields": "Method", "session_id": "x"}).json()
    assert r["fields"] == ["Method"] and r["rows"][0]["Method"] == "Cash on delivery"
    assert r["rows"][0]["source"]["url"].startswith("https://atlas-home.example/")
    csv_text = client.get("/api/kb/ex-shop.csv").text
    assert "Cash on delivery" in csv_text and "'=HYPERLINK" in csv_text
    js = client.get("/api/kb/ex-shop.json").json()
    assert js["rows"] and js["pages_text"][0]["text"]
    assert client.get("/api/kb/nope.json").status_code == 404


def test_fake_forwarded_for_does_not_dodge_ip_cap(monkeypatch):
    monkeypatch.delenv("TRUSTED_PROXY_HOPS", raising=False)
    monkeypatch.delenv("CLIENT_IP_HEADER", raising=False)
    client, fake, _ = make(budget=Budget(session_cap=99, ip_cap=1))

    def ask_as(fake_ip, session, q):  # our proxy appends the real address (203.0.113.7) after what the visitor sent
        r = client.post("/api/chat", headers={"X-Forwarded-For": f"{fake_ip}, 203.0.113.7"},
                        json={"kb_id": "ex-shop", "session_id": session, "messages": [{"role": "user", "content": q}]})
        return events(r)

    ask_as("1.1.1.1", "a", "Can I pay cash on delivery?")
    ev = ask_as("9.9.9.9", "b", "How long does delivery to Casablanca take?")  # new session id AND a new fake address
    assert ev[2] == ("notice", {"code": "ip_cap", "message": ev[2][1]["message"]})
    assert fake.calls == 1


def test_client_ip_settings(monkeypatch):
    from starlette.requests import Request

    def req(headers, peer="10.0.0.1"):
        raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
        return Request({"type": "http", "headers": raw, "client": (peer, 1234)})

    monkeypatch.delenv("CLIENT_IP_HEADER", raising=False)
    monkeypatch.delenv("TRUSTED_PROXY_HOPS", raising=False)
    assert main.client_ip(req({})) == "10.0.0.1"
    assert main.client_ip(req({"X-Forwarded-For": "6.6.6.6, 5.5.5.5"})) == "5.5.5.5"
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "2")
    assert main.client_ip(req({"X-Forwarded-For": "6.6.6.6, 5.5.5.5, 4.4.4.4"})) == "5.5.5.5"
    assert main.client_ip(req({"X-Forwarded-For": "5.5.5.5"})) == "5.5.5.5"
    monkeypatch.setenv("CLIENT_IP_HEADER", "cf-connecting-ip")
    assert main.client_ip(req({"CF-Connecting-IP": "7.7.7.7", "X-Forwarded-For": "6.6.6.6"})) == "7.7.7.7"


def test_stream_crawl_is_counted_before_the_stream_runs():
    calls = []

    def crawl(url, max_pages=15, on_page=None, **kw):
        calls.append(url)
        return scraper.CrawlReport(url, [scraper.Page(url, "T", "word " * 50)], 1, "done", False, [])

    client, _, app = make(crawl=crawl, budget=Budget(ip_crawl_cap=1))
    first = client.post("/api/kb?stream=1", json={"url": "c.example"})
    assert first.status_code == 200 and app.state.budget.snapshot()["crawls"] == 1
    r = client.post("/api/kb?stream=1", json={"url": "c.example"})
    assert r.status_code == 429 and r.json()["error"]["code"] == "ip_crawl_cap" and len(calls) == 1


def test_odd_bodies_get_friendly_errors_not_500():
    client, _, _ = make()
    for path in ("/api/kb", "/api/extract"):
        for body in ([1, 2], "x", {"url": "a.example", "max_pages": "lots"}):
            r = client.post(path, json=body)
            assert r.status_code in (400, 404, 413, 422) and "error" in r.json(), (path, body)
    big = client.post("/api/kb", content=b"{" + b" " * 300_000 + b"}", headers={"Content-Type": "application/json"})
    assert big.status_code == 413
    huge = client.post("/api/kb", content=b"x" * 30, headers={"Content-Type": "multipart/form-data; boundary=x",
                                                               "Content-Length": str(25_000_000)})
    assert huge.status_code == 413


def test_long_history_is_cut_before_search():
    seen = {}
    client, fake, app = make()
    kb = app.state.kbs.get("ex-shop")
    real = kb.index.search

    def spy(question, *a, prev_question=None, **kw):
        seen["prev"] = prev_question
        return real(question, *a, prev_question=prev_question, **kw)

    kb.index.search = spy
    ev = ask(client, "and cash?", history=[{"role": "user", "content": "delivery " * 2_000},
                                           {"role": "assistant", "content": "ok"}])
    assert ev[-1][0] == "done", ev
    assert len(seen["prev"]) <= main.MAX_HISTORY_CHARS


def _fake_pdf_pages(monkeypatch, n=2):
    calls = []

    def read_pdfs(files):
        calls.append([f.name for f in files])
        return [scraper.Page(f"{files[0].name}#page={i}", f"Page {i}", "Refunds take five working days. " * 10)
                for i in range(1, n + 1)], []
    monkeypatch.setattr(main.docs, "read_pdfs", read_pdfs)
    return calls


def test_pdf_upload_cap_per_ip(monkeypatch):
    monkeypatch.delenv("TRUSTED_PROXY_HOPS", raising=False)
    monkeypatch.delenv("CLIENT_IP_HEADER", raising=False)
    calls = _fake_pdf_pages(monkeypatch)
    client, _, app = make(budget=Budget(ip_upload_cap=2))

    def upload(fake_ip):  # the visitor changes the address they send; our proxy appends the real one
        return client.post("/api/kb", headers={"X-Forwarded-For": f"{fake_ip}, 203.0.113.7"},
                           files={"files": ("guide.pdf", b"%PDF-1.4 fake", "application/pdf")})

    assert upload("1.1.1.1").status_code == 200 and upload("2.2.2.2").status_code == 200
    r = upload("3.3.3.3")
    assert r.status_code == 429 and r.json()["error"]["code"] == "ip_upload_cap"
    assert "PDF" in r.json()["error"]["message"] and len(calls) == 2
    assert app.state.budget.snapshot()["uploads"] == 2
    r = client.post("/api/kb?stream=1", headers={"X-Forwarded-For": "203.0.113.7"},
                    files={"files": ("guide.pdf", b"%PDF", "application/pdf")})
    assert r.status_code == 429 and r.json()["error"]["code"] == "ip_upload_cap"
    other = client.post("/api/kb", headers={"X-Forwarded-For": "198.51.100.9"},
                        files={"files": ("guide.pdf", b"%PDF", "application/pdf")})
    assert other.status_code == 200 and len(calls) == 3  # another network still can upload


def test_ip_upload_cap_env(monkeypatch):
    monkeypatch.setenv("IP_UPLOAD_CAP", "1")
    _, _, app = make()
    assert app.state.budget.ip_upload_cap == 1
    monkeypatch.delenv("IP_UPLOAD_CAP")
    _, _, app = make()
    assert app.state.budget.ip_upload_cap == 10


def test_page_list_in_kb_json_stream_done_pdf_and_examples(monkeypatch):
    def crawl(url, max_pages=15, on_page=None, **kw):
        pages = [scraper.Page(f"{url}p{i}", f"Title {i}", "word " * 50) for i in range(25)]
        for i, p in enumerate(pages, 1):
            if on_page:
                on_page(i, p.url)
        return scraper.CrawlReport(url, pages, 25, "done", False, [])

    client, _, _ = make(crawl=crawl, budget=Budget(ip_crawl_cap=9))
    js = client.post("/api/kb", json={"url": "c.example"}).json()
    assert len(js["page_list"]) == 20 and js["pages"] == 25
    assert js["page_list"][0] == {"title": "Title 0", "url": "https://c.example/p0", "rendered": False}
    assert [p["title"] for p in js["page_list"]] == [f"Title {i}" for i in range(20)]  # crawl order
    done = [d for e, d in events(client.post("/api/kb?stream=1", json={"url": "c.example"})) if e == "done"][0]
    assert done["page_list"] == js["page_list"]
    _fake_pdf_pages(monkeypatch, n=3)
    pdf = client.post("/api/kb", files={"files": ("guide.pdf", b"%PDF", "application/pdf")}).json()
    assert [p["url"] for p in pdf["page_list"]] == ["guide.pdf#page=1", "guide.pdf#page=2", "guide.pdf#page=3"]
    for ex in client.get("/api/examples").json()["examples"]:
        assert ex["page_list"] and set(ex["page_list"][0]) == {"title", "url", "rendered"}


def test_sources_hl_at_is_a_javascript_span_of_hl():
    client, _, _ = make()
    for kb, q in (("ex-shop", "How long does delivery to Casablanca take?"), ("ex-shop_ar", "كم تكلفة الشحن؟")):
        sources = dict(ask(client, q, kb=kb, session=kb))["sources"]
        assert sources
        for s in sources:
            if not s["hl"]:
                assert s["hl_at"] is None
                continue
            a, b = s["hl_at"]
            utf16 = s["passage"].encode("utf-16-le")
            assert utf16[2 * a:2 * b].decode("utf-16-le") == s["hl"]


def test_js_span_counts_utf16_units():
    assert main._js_span("😀 ab", "ab") == [3, 5]  # the emoji is 2 JavaScript units
    assert main._js_span("abc", "") is None and main._js_span("abc", "zz") is None


def test_arabic_example_has_four_starters():
    client, _, _ = make()
    ar = next(e for e in client.get("/api/examples").json()["examples"] if e["kb_id"] == "ex-shop_ar")
    assert len(ar["starters"]) == 4 and ar["lang"] == "ar"


def test_export_examples_keeps_the_live_stream_shape():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import export_examples as exp
    client, fake, _ = make(budget=Budget(session_cap=99, ip_cap=99))
    data, problems = exp.build(client, log=lambda *a: None, followups=False, extract=False)
    assert not problems and fake.calls == sum(len(e["starters"]) for e in data["examples"])
    shop = next(e for e in data["examples"] if e["kb_id"] == "ex-shop")
    assert set(shop) == {"kb_id", "label", "kind", "url", "lang", "pages", "words", "page_list", "starters"}
    a = shop["starters"][0]
    live = dict(ask(make()[0], a["q"]))
    assert set(a) == {"q", "meta", "sources", "notice", "text", "followups", "done"}
    assert a["meta"] == live["meta"] and a["sources"] == live["sources"] and a["followups"] == live["followups"]
    assert set(a["done"]) == set(live["done"]) and a["done"]["left_today"] is None and a["done"]["status"] == "answered"
    assert a["text"].startswith("Delivery to Casablanca") and "FOLLOWUPS" not in a["text"]
    with pytest.raises(RuntimeError):
        exp.to_answer("q", [("error", {"code": "kb_expired", "message": "x"})])


# ---------------------------------------------------------------- wave 3 fixes

def test_done_hl_points_at_the_sentence_the_answer_used():
    fake = FakeLLM("A return fee of 25 MAD is deducted from your refund [1].\nFOLLOWUPS: How long do refunds take?")
    client, _, _ = make(stream=fake)
    history = [{"role": "user", "content": "How many days do I have to return an item?"},
               {"role": "assistant", "content": "You have 14 days from delivery [1]."}]
    ev = dict(ask(client, "How much is the return fee?", history=history))
    src = ev["sources"][0]
    # the guess uses this question's words, not the earlier "return an item within 14 days" question
    assert "25 MAD return fee" in src["hl"]
    hl = ev["done"]["hl"]["1"]
    assert hl["hl"] == "Otherwise a 25 MAD return fee is deducted from your refund."
    a, b = hl["hl_at"]
    assert src["passage"][a:b] == hl["hl"]  # ASCII text: JS indices equal Python indices
    assert ev["followups"] == ["How long do refunds take?"]


def test_cached_arabic_answer_highlights_its_own_sentence_at_once():
    fake = FakeLLM("يستغرق التوصيل من 2 إلى 4 أيام عمل [1].")
    client, _, _ = make(stream=fake)
    q = "كم يستغرق التوصيل؟"
    first = dict(ask(client, q, kb="ex-shop_ar"))
    assert first["done"]["hl"]["1"]["hl"].startswith("نوصل الطلبات")
    again = dict(ask(client, q, kb="ex-shop_ar", session="s2"))
    assert again["done"]["cached"] is True and fake.calls == 1
    assert again["sources"][0]["hl"].startswith("نوصل الطلبات")  # no wait for done on a cached answer
    a, b = again["sources"][0]["hl_at"]
    assert again["sources"][0]["passage"].encode("utf-16-le")[2 * a:2 * b].decode("utf-16-le") == again["sources"][0]["hl"]


def test_followups_the_site_cannot_answer_are_dropped():
    fake = FakeLLM("Delivery takes 1 to 2 days [1].\nFOLLOWUPS: Can I track it? | Who won the 2022 World Cup? | Is it free?")
    client, _, _ = make(stream=fake)
    ev = dict(ask(client, "How long does delivery to Casablanca take?"))
    assert ev["followups"] == ["Can I track it?", "Is it free?"]


def test_chunked_upload_over_the_limit_is_stopped_while_streaming(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    client, _, _ = make()
    boundary = "xyz"
    head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"a.pdf\"\r\n"
            "Content-Type: application/pdf\r\n\r\n").encode()
    sent = {"n": 0}

    def body():  # a generator body: httpx sends it chunked, with no Content-Length
        yield head
        for _ in range(40):  # 40 x 64 KB = 2.6 MB
            sent["n"] += 1
            yield b"%" * 65_536
        yield f"\r\n--{boundary}--\r\n".encode()

    r = client.post("/api/kb", content=body(), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert r.status_code == 413 and r.json()["error"]["code"] == "too_big"
    small = client.post("/api/kb", files={"files": ("a.pdf", b"%PDF-1.4 tiny", "application/pdf")})
    assert small.status_code != 413


def test_extract_cap_warning_does_not_promise_passages():
    def complete(messages, schema, name="result"):
        return {"rows": []}, "fake/model"

    client, _, _ = make(complete=complete, budget=Budget(session_cap=0, ip_cap=30))
    r = client.post("/api/extract", json={"kb_id": "ex-shop", "fields": "Method", "session_id": "x"}).json()
    assert r["rows"] == [] and r["warnings"] and "passages" not in r["warnings"][0]
    assert "extraction stopped" in r["warnings"][0]


def test_export_folds_done_hl_into_sources():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import export_examples as exp
    ans = exp.to_answer("q", [("sources", [{"n": 1, "passage": "A. B.", "hl": "A.", "hl_at": [0, 2]}]),
                              ("delta", {"text": "B [1]"}),
                              ("done", {"status": "answered", "hl": {"1": {"hl": "B.", "hl_at": [3, 5]}}})])
    assert ans["sources"][0]["hl"] == "B." and ans["sources"][0]["hl_at"] == [3, 5]


def test_fastapi_example_has_no_terminal_markup():
    client, fake, _ = make()
    ev = dict(ask(client, "Which command starts the development server?", kb="ex-fastapi_tutorial"))
    assert ev["sources"] and not any("<font" in s["passage"] or "<span" in s["passage"] for s in ev["sources"])


def test_export_adds_followup_answers_and_extract_tables():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import export_examples as exp
    rows = [{"fee or rule": "Shipping", "amount": "35 MAD", "details": "free over 500 MAD"}]
    client, fake, _ = make(budget=Budget(session_cap=999, ip_cap=999),
                           complete=lambda m, schema, name="result": ({"rows": rows}, "fake/model"))
    data, problems = exp.build(client, log=lambda *a: None)
    shop = next(e for e in data["examples"] if e["kb_id"] == "ex-shop")
    assert set(shop["followup_answers"]) == {"Is it free?", "Can I track it?"}
    a = shop["followup_answers"]["Is it free?"]
    assert set(a) == {"q", "meta", "sources", "notice", "text", "followups", "done"}
    assert shop["extract"]["fields"] == ["fee or rule", "amount", "details"] and shop["extract"]["rows"]
    assert next(e for e in data["examples"] if e["kb_id"] == "ex-python_tutorial")["extract"] is None
