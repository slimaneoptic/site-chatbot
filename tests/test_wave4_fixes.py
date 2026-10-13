"""Wave 4 review fixes: hostile-input speed (regex), Origin guard, renderer limits, dropped-model cost,
translated-copy filter, translation hint, parallel start render. No network."""
import json
import time

import pytest

import llm
import scraper
import search
from scraper import Page, read_site
from test_llm import MSGS, FakeResp, fake_post, sse  # noqa: F401 - fake_post is a fixture
from test_scraper import CRA, WORDS, Resp, html, page, site  # noqa: F401 - `site` is a fixture
from test_scraper_js import J, SITEMAP, jina, js_routes


def _fast(fn, *a, limit=0.5):
    t = time.perf_counter()
    out = fn(*a)
    took = time.perf_counter() - t
    assert took < limit, f"{fn.__name__} took {took:.2f} s"
    return out


# ---- HIGH 1-3: no quadratic regex on hostile pages -------------------------------------------------

def test_site_info_lines_long_whitespace_is_linear():
    doc = "<html><body><p>hello</p><footer>Call us\n   " + " " * 200_000 + "b\n  Rabat </footer></body></html>"
    lines = _fast(scraper._site_info_lines, doc)
    assert "Call us" in lines and "Rabat" in lines


def test_sitemap_unclosed_tags_are_linear_and_real_sitemaps_still_parse(monkeypatch):
    class Robots:
        def site_maps(self):
            return []
    bodies = {}
    monkeypatch.setattr(scraper, "_download", lambda s, url, **k: (url, 200, {}, bodies[url], "utf-8"))
    for hostile in ("<sitemapindex>" + "<loc>" * 100_000, "<urlset>" + "<url>" * 100_000, "<urlset><url><loc>" + "a" * 400_000):
        bodies["https://e.test/sitemap.xml"] = hostile.encode()
        _fast(scraper._sitemap_urls, None, "https", "e.test", Robots(), time.monotonic() + 60)
    bodies["https://e.test/sitemap.xml"] = SITEMAP.replace("app.test", "e.test").replace("<loc>", "<loc>\n  ").encode()
    got = scraper._sitemap_urls(None, "https", "e.test", Robots(), time.monotonic() + 60)
    assert got[:2] == [("https://e.test/", 1.0), ("https://e.test/simulateur", 0.9)] and len(got) == 4
    # an index points to child sitemaps (3 at most, same host)
    bodies["https://e.test/sitemap.xml"] = (b"<sitemapindex><sitemap><loc> https://e.test/a.xml </loc></sitemap>"
                                            b"<sitemap><loc>https://other.test/b.xml</loc></sitemap></sitemapindex>")
    bodies["https://e.test/a.xml"] = b"<urlset><url><loc>https://e.test/x?a=1&amp;b=2</loc></url></urlset>"
    assert scraper._sitemap_urls(None, "https", "e.test", Robots(), time.monotonic() + 60) == [("https://e.test/x?a=1&b=2", 0.5)]


def test_sitemap_reads_at_most_the_block_cap(monkeypatch):
    class Robots:
        def site_maps(self):
            return []
    xml = "<urlset>" + "".join(f"<url><loc>https://e.test/p{i}</loc></url>" for i in range(1000)) + "</urlset>"
    monkeypatch.setattr(scraper, "_download", lambda s, url, **k: (url, 200, {}, xml.encode(), "utf-8"))
    assert len(scraper._sitemap_urls(None, "https", "e.test", Robots(), time.monotonic() + 60)) == scraper.SITEMAP_MAX_BLOCKS


@pytest.mark.parametrize("line", ["[" * 40_000, "\\[" * 20_000, "[a](xx" * 7_000, "![" * 20_000, "[a](" + "x" * 19_000,
                                  "[" * 30_000 + "](x)"])
def test_markdown_hostile_lines_are_fast(line):
    _fast(scraper.markdown_text, line)
    _fast(scraper.markdown_links, line, "https://a.example/")


def test_markdown_still_parses_links_images_and_drops_giant_lines():
    md = ('# Title\n![logo](https://a.example/l.png "Logo") [Prices](/prices "Our prices") and [Contact](https://a.example/c)\n'
          + "x" * 25_000 + " [Hidden](https://a.example/h)\n")
    assert scraper.markdown_links(md, "https://a.example/") == [("https://a.example/prices", "Prices"),
                                                               ("https://a.example/c", "Contact")]
    text = scraper.markdown_text(md)
    assert text.startswith("Title\n Prices and Contact") and "Hidden" not in text


def test_renderer_body_is_capped(site, monkeypatch):
    seen = {}
    real = scraper._download

    def spy(session, url, **kw):
        if url.startswith(J):
            seen.update(kw)
        return real(session, url, **kw)
    monkeypatch.setattr(scraper, "_download", spy)
    site(js_routes())
    read_site("app.test", 2)
    assert seen["max_bytes"] == scraper.JINA_MAX_BYTES == 300_000 and seen["truncate"] is True


def test_whole_read_of_a_hostile_2mb_page_is_fast(site):
    junk = ("<div class='footer'>a" + " " * 300_000 + "b</div>" + "[" * 100_000 + "<url><loc>" * 20_000
            + "<a href='/x'>" * 2_000 + " " * 1_350_000)
    body = page(WORDS, "Home", ["/a", "/b"]).replace("</body>", junk + "</body>")
    assert 1_900_000 < len(body) < scraper.MAX_BYTES
    routes = {"https://h.test/robots.txt": Resp(404, b""), "https://h.test/": html(body),
              "https://h.test/sitemap.xml": Resp(200, "<urlset>" + "<url><loc>" * 100_000, {"Content-Type": "application/xml"}),
              J + "https://h.test/": jina("H", "https://h.test/", "[" * 290_000)}
    site(routes)
    t = time.perf_counter()
    report = read_site("https://h.test/", 5)
    assert time.perf_counter() - t < 3 and report.pages and report.pages[0].url == "https://h.test/"


# ---- MEDIUM 4: another site's page can't POST to the API ---------------------------------------------

def test_post_from_a_foreign_origin_is_refused_before_any_work(monkeypatch):
    from fastapi.testclient import TestClient
    from api import main
    from test_api import FakeLLM
    monkeypatch.setenv("CORS_ORIGINS", "https://slimanechanchoul.com")
    stream = FakeLLM()
    app = main.create_app(stream=stream, check_key=False)
    c = TestClient(app)
    ex = c.get("/api/examples").json()["examples"][0]
    body = json.dumps({"kb_id": ex["kb_id"], "messages": [{"role": "user", "content": ex["starters"][0]["q"] + " why?"}],
                       "session_id": "s1"})
    for path, payload in (("/api/chat", body), ("/api/kb", json.dumps({"url": "https://b.example/"})),
                          ("/api/extract", json.dumps({"kb_id": ex["kb_id"], "fields": ["price"]}))):
        for origin in ("https://evil.example", "null", "https://slimanechanchoul.com.evil.example"):
            r = c.post(path, content=payload, headers={"Content-Type": "text/plain", "Origin": origin})
            assert r.status_code == 403 and r.json()["error"]["code"] == "origin_not_allowed", (path, origin)
    snap = app.state.budget.snapshot()
    assert stream.calls == 0
    assert snap["questions"] == 0 and snap["crawls"] == 0 and snap["spent_usd"] == 0
    for origin in ("https://slimanechanchoul.com", "http://localhost:8080", None):  # the real pages, and no Origin
        headers = {"Content-Type": "text/plain"} | ({"Origin": origin} if origin else {})
        r = c.post("/api/chat", content=body, headers=headers)
        assert r.status_code == 200 and "event: done" in r.text
    assert c.get("/api/examples", headers={"Origin": "https://evil.example"}).status_code == 200  # GET is not guarded


# ---- LOW 5-7: renderer limits and a finished read ------------------------------------------------------

def test_renderer_calls_stop_at_twice_the_page_cap(monkeypatch):
    monkeypatch.setattr(scraper, "_JINA_BUCKET", scraper._Bucket())
    monkeypatch.setattr(scraper, "check_public_url", lambda u: None)
    monkeypatch.setenv("JS_MAX_PAGES", "2")
    run = scraper._Run()
    r = scraper._Renderer(run, time.monotonic() + 30)
    asked = []
    monkeypatch.setattr(r, "_get", lambda url, fresh: (asked.append(url) or 200, b"Markdown Content:\nthin"))
    for i in range(10):
        assert r.render(f"https://a.test/{i}") is None
    assert run.jina_calls <= 2 * r.max_pages + 1 and len(asked) <= 2 * r.max_pages * 2


def test_start_page_that_lands_on_a_robots_blocked_path_is_never_rendered(site):
    sess = site({
        "https://app.test/robots.txt": Resp(200, "User-agent: *\nDisallow: /private/\n", {"Content-Type": "text/plain"}),
        "https://app.test/": Resp(302, b"", {"Location": "https://app.test/private/app"}),
        "https://app.test/private/app": html(CRA),
        J + "https://app.test/private/app": jina("P", "https://app.test/private/app", "secret words " * 50),
    })
    with pytest.raises(scraper.CrawlError) as info:
        read_site("https://app.test/")
    assert info.value.code == "robots" and not any(c.startswith(J) for c in sess.calls)


def test_closed_read_stops_queued_work_and_sessions_exist_up_front(monkeypatch):
    made = []
    monkeypatch.setattr(scraper.requests, "Session", lambda: made.append(1) or type("S", (), {"headers": {}, "close": lambda s: None})())
    run = scraper._Run()
    r = scraper._Renderer(run, time.monotonic() + 30)
    assert len(made) == 2 and r._sess(False) is not r._sess(True)
    run.closed = True
    assert r.render("https://a.test/") is None and run.jina_calls == 0
    got = scraper._fetch_one(None, r, "https://a.test/", first=False, via_render=True, deadline=time.monotonic() + 5,
                             allow=None)
    assert got.error is not None and got.text == "" and len(made) == 2


def test_read_marks_the_run_closed(site, monkeypatch):
    runs = []
    real = scraper._Run
    monkeypatch.setattr(scraper, "_Run", lambda: runs.append(real()) or runs[-1])
    site(js_routes())
    read_site("app.test", 2)
    assert runs and all(r.closed for r in runs)


# ---- LOW 8: a model dropped for slow first text is still paid for ----------------------------------------

def test_model_dropped_after_first_token_wait_counts_its_cost(fake_post, monkeypatch):
    _, replies = fake_post
    monkeypatch.setattr(llm, "FIRST_TOKEN_S", 0.0)
    monkeypatch.setattr(llm, "PRICES", {"m1": (1.0, 2.0), "m2": (1.0, 2.0)})
    replies += [FakeResp(lines=[": keep-alive", ": keep-alive"] + sse("late")),
                FakeResp(lines=sse("ok", usage={"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.001}))]
    seen = []
    assert "".join(llm.stream_text(MSGS, on_usage=seen.append)) == "ok"
    dropped = llm.estimate_cost("m1", 100, llm.DROPPED_THINK_TOKENS)
    assert len(seen) == 1 and seen[0].calls == 2 and seen[0].model == "m2"
    assert seen[0].cost_usd == pytest.approx(0.001 + dropped)


def test_dropped_model_cost_is_reported_even_when_no_model_answers(fake_post, monkeypatch):
    _, replies = fake_post
    monkeypatch.setattr(llm, "FIRST_TOKEN_S", 0.0)
    monkeypatch.setattr(llm, "PRICES", {"m1": (1.0, 2.0)})
    replies += [FakeResp(lines=[": keep-alive"] + sse("late")), FakeResp(503)]
    seen = []
    with pytest.raises(llm.LLMError):
        "".join(llm.stream_text(MSGS, on_usage=seen.append))
    assert len(seen) == 1 and seen[0].calls == 1 and seen[0].cost_usd > 0


# ---- A: translated copies don't fill the read -------------------------------------------------------------

def test_other_language_folders_and_hreflang_copies_are_skipped(site):
    home = page(WORDS, "Home", ["/about/", "/de/about/", "/pt-br/", "/zh-hant/docs/", "/en/guide/", "/es/", "/docs/",
                                "/it-is-fine/", "/blog/"])
    home = home.replace("<html>", '<html lang="en">')
    docs = page(WORDS + " docs", "Docs", ["/docs/x"]).replace(
        "<head>", '<head><link rel="alternate" hreflang="fr" href="/docs-fr/"><link rel="alternate" hreflang="en" href="/docs/">')
    routes = {"https://s.test/robots.txt": Resp(404, b""), "https://s.test/": html(home), "https://s.test/docs/": html(docs)}
    for path in ("about/", "de/about/", "pt-br/", "zh-hant/docs/", "en/guide/", "es/", "docs-fr/", "docs/x", "it-is-fine/", "blog/"):
        routes[f"https://s.test/{path}"] = html(page(WORDS + " " + path, path))
    sess = site(routes)
    urls = [p.url for p in read_site("https://s.test/", 15).pages]
    assert {"https://s.test/about/", "https://s.test/en/guide/", "https://s.test/docs/", "https://s.test/it-is-fine/"} <= set(urls)
    for skipped in ("de/about/", "pt-br/", "zh-hant/docs/", "es/"):
        assert f"https://s.test/{skipped}" not in sess.calls
    assert "https://s.test/docs-fr/" not in urls


def test_a_start_page_inside_a_language_folder_keeps_its_own_folder(site):
    start = page(WORDS, "Accueil", ["/fr/a/", "/en/a/", "/fr/contact/"]).replace("<html>", '<html lang="fr">')
    routes = {"https://s.test/robots.txt": Resp(404, b""), "https://s.test/fr/": html(start)}
    for path in ("fr/a/", "en/a/", "fr/contact/"):
        routes[f"https://s.test/{path}"] = html(page(WORDS + path, path))
    site(routes)
    urls = [p.url for p in read_site("https://s.test/fr/", 10).pages]
    assert "https://s.test/fr/a/" in urls and "https://s.test/fr/contact/" in urls and "https://s.test/en/a/" not in urls


def test_unknown_start_language_skips_nothing(site):
    routes = {"https://s.test/robots.txt": Resp(404, b""), "https://s.test/": html(page(WORDS, "Home", ["/de/", "/fr/"]))}
    for path in ("de/", "fr/"):
        routes[f"https://s.test/{path}"] = html(page(WORDS + path, path))
    site(routes)
    assert len([p for p in read_site("https://s.test/", 10).pages if "#" not in p.url]) == 3


# ---- B: a question in another language uses the site's own words -------------------------------------------

def test_translation_gets_the_site_words_and_finds_the_rate():
    pages = [Page("https://e.test/", "Crédit rapide | Eqdom",
                  "Votre crédit en un clic. Un crédit pensé pour vos projets avec un taux à partir de 7.5% et une "
                  "réponse rapide. Simulez votre crédit en ligne."),
             Page("https://e.test/contact", "Nous contacter", "Contactez nos agences pour votre demande de crédit. " * 3)]
    asked = []

    def tr(text, lang, hint=""):
        asked.append((lang, hint))
        return "Quel est le taux du crédit ?" if "crédit" in hint else "Quel est le taux d'intérêt du prêt ?"
    found = search.Index(pages).search("ما هي نسبة الفائدة على القرض؟", translate=tr)
    assert asked and asked[0][0] == "fr" and "crédit" in asked[0][1] and "Crédit rapide | Eqdom" in asked[0][1]
    assert found.translated and found.matched and "7.5%" in found.passages[0].text and found.lang == "ar"


def test_translate_query_sends_the_hint(fake_post):
    calls, replies = fake_post
    replies.append(FakeResp(text=json.dumps({"choices": [{"message": {"content": "Quel est le taux du crédit ?"}}]})))
    assert llm.translate_query("ما هي نسبة الفائدة؟", "fr", hint="crédit, taux") == "Quel est le taux du crédit ?"
    assert "crédit, taux" in calls[0]["messages"][0]["content"]


# ---- C: a JavaScript site renders its start page alongside the first pages, and stops at 6 when slow -----------

def test_js_start_page_renders_in_parallel_with_the_first_pages(site, monkeypatch):
    monkeypatch.setattr(scraper, "FETCH_WORKERS", 4)
    routes = js_routes()
    active, peak = [0], [0]
    lock = __import__("threading").Lock()

    def slow(resp):
        def get():
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.2)
            with lock:
                active[0] -= 1
            return resp
        return get
    for k in list(routes):
        if k.startswith(J):
            routes[k] = slow(routes[k])
    site(routes)
    t = time.perf_counter()
    report = read_site("app.test", 5)
    assert report.pages[0].url == "https://app.test/" and report.pages[0].rendered
    assert peak[0] >= 3 and time.perf_counter() - t < 0.2 * 4


def test_slow_js_site_stops_at_six_pages_after_the_soft_limit(site, monkeypatch):
    monkeypatch.setattr(scraper, "JS_SOFT_S", 0)
    extra = tuple(f"p{i}" for i in range(10))
    routes = js_routes(extra)
    routes["https://app.test/sitemap.xml"] = Resp(200, "<urlset>" + "".join(
        f"<url><loc>https://app.test/{p}</loc></url>" for p in ("simulateur", "nous-contacter", "foire-aux-questions") + extra)
        + "</urlset>", {"Content-Type": "application/xml"})
    for p in extra:
        routes[J + f"https://app.test/{p}"] = jina(p, f"https://app.test/{p}", f"# {p}\n\n" + f"Page {p} text about credit. " * 12)
    monkeypatch.setenv("JS_MAX_PAGES", "20")
    site(routes)
    report = read_site("app.test", 15)
    assert len([p for p in report.pages if "#" not in p.url]) == scraper.JS_SOFT_PAGES and report.stopped == "js_time"


def test_long_passage_keeps_the_rate_lines_for_a_translated_rate_question():
    text = ("Votre crédit en un clic !\nSimulez votre crédit\nBesoin d’assistance crédit ?\n"
            "Je vous aide à préparer et comprendre votre demande de crédit, à chaque étape, avec des réponses rapides "
            "et claires pour tous vos projets de crédit.\nSalarié du privé\nUn crédit pensé pour vos projets du quotidien. "
            "Financement adapté à votre situation.\nÀ partir de 7.5%\nVoir l'offre\nFonctionnaire\nConditions avantageuses "
            "via notre partenariat Fondation. Taux réduits et traitement prioritaire.\nÀ partir de 5.5%\nRetraité\n"
            "Une solution claire et adaptée à votre budget. Mensualités maîtrisées pour vos projets de crédit.\n"
            "À partir de 6.90%\nProfessionnel\nFinancez votre activité en toute sérénité avec un crédit adapté.")
    p = Page("u", "t", text)
    out = search.trim_passage(search.Passage("u", "t", text), "ما هي نسبة الفائدة على القرض؟ Quel est le taux d'intérêt pour un crédit ?",
                              max_chars=420)
    assert len(out.text) <= 420 and "7.5%" in out.text and "5.5%" in out.text and "Taux réduits" in out.text
    assert p.words > 0
