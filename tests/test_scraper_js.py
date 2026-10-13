"""JavaScript-site fallback (Jina Reader, faked), sitemap seeding, parallel fetch, crawl priority, titles.
No network: the fake session from test_scraper serves both the site and the renderer."""
import threading
import time

import pytest

import scraper
from scraper import CrawlError, read_site
from test_scraper import CRA, WORDS, FakeSession, Resp, fake_dns, html, page, site  # noqa: F401 - `site` is a fixture

J = "https://r.jina.ai/"


def jina(title, source, md, status=200):
    return Resp(status, f"Title: {title}\n\nURL Source: {source}\n\nMarkdown Content:\n{md}", {"Content-Type": "text/plain"})


RATE_MD = """## Votre crédit en un clic !

![Image 1](https://app.test/static/a.webp)

### Salarié du privé

Un crédit pensé pour vos projets du quotidien. Financement adapté à votre situation professionnelle, avec un
taux À partir de **7.5%** et une réponse rapide. Simulez votre crédit en ligne et déposez votre demande en quelques
minutes depuis votre téléphone, sans vous déplacer en agence.

[Voir l'offre →](https://app.test/offre) [Nous contacter](/nous-contacter) [Logo](https://other.test/x)
"""

SITEMAP = """<?xml version="1.0"?><urlset>
<url><loc>https://app.test/</loc><priority>1.0</priority></url>
<url><loc>https://app.test/simulateur</loc><priority>0.9</priority></url>
<url><loc>https://app.test/nous-contacter</loc><priority>0.7</priority></url>
<url><loc>https://app.test/foire-aux-questions</loc><priority>0.6</priority></url>
</urlset>"""


def js_routes(extra_pages=()):
    r = {
        "https://app.test/robots.txt": Resp(200, "User-agent: *\nAllow: /\nSitemap: https://app.test/sitemap.xml\n",
                                            {"Content-Type": "text/plain"}),
        "https://app.test/": html(CRA),
        "https://app.test/sitemap.xml": Resp(200, SITEMAP, {"Content-Type": "application/xml"}),
        J + "https://app.test/": jina("Crédit rapide | Eqdom", "https://app.test/", RATE_MD),
    }
    for path in ("simulateur", "nous-contacter", "foire-aux-questions", "offre") + tuple(extra_pages):
        r[f"https://app.test/{path}"] = html(CRA)
        r[J + f"https://app.test/{path}"] = jina(f"{path} | Eqdom", f"https://app.test/{path}",
                                                 f"# {path}\n\n" + "Les questions fréquentes sur le crédit et la simulation. " * 8)
    return r


def test_js_site_is_read_through_the_renderer_with_sitemap_seeding(site):
    sess = site(js_routes())
    report = read_site("app.test")
    urls = [p.url for p in report.pages]
    assert urls[0] == "https://app.test/" and len(report.pages) >= 4
    assert {"https://app.test/simulateur", "https://app.test/nous-contacter", "https://app.test/foire-aux-questions"} <= set(urls)
    assert all(p.rendered for p in report.pages if not p.url.endswith("#site-info"))
    assert "7.5%" in report.pages[0].text and "**" not in report.pages[0].text and "webp" not in report.pages[0].text
    assert report.rendered >= 4 and report.jina_calls >= 4 and report.pages[0].title == "Crédit rapide"
    assert any("read through a page renderer" in w for w in report.warnings)
    # sitemap pages of a JS site go straight to the renderer (no wasted HTML fetch)
    assert "https://app.test/simulateur" not in sess.calls and J + "https://app.test/simulateur" in sess.calls
    assert not any("other.test" in c for c in sess.calls)


def test_server_rendered_site_makes_no_renderer_call(site):
    from test_scraper import shop_routes
    sess = site(shop_routes())
    report = read_site("https://shop.test/help/")
    assert report.jina_calls == 0 and not any("jina" in c for c in sess.calls) and len(report.pages) >= 3


@pytest.mark.parametrize("target", ["http://10.0.0.1/", "http://localhost/", "http://intranet.test/app/"])
def test_internal_targets_never_reach_the_renderer(site, target, monkeypatch):
    sess = site({})
    monkeypatch.setattr("socket.getaddrinfo", lambda h, *a, **k: fake_dns("127.0.0.1" if h == "localhost" else h))
    run = scraper._Run()
    assert scraper._Renderer(run, time.monotonic() + 30).render(target) is None
    with pytest.raises(CrawlError):
        read_site(target)
    assert not any("jina" in c for c in sess.calls) and run.jina_calls == 0


def test_renderer_page_cap(site, monkeypatch):
    monkeypatch.setenv("JS_MAX_PAGES", "8")
    extra = [f"p{i}" for i in range(12)]
    routes = js_routes(extra)
    routes["https://app.test/sitemap.xml"] = Resp(200, "<urlset>" + "".join(
        f"<url><loc>https://app.test/{p}</loc></url>" for p in extra) + "</urlset>", {"Content-Type": "application/xml"})
    site(routes)
    report = read_site("app.test", 15)
    assert report.rendered == 8 and sum(p.rendered for p in report.pages) == 8


def test_renderer_429_stops_rendering_for_this_read(site):
    routes = js_routes()
    routes[J + "https://app.test/simulateur"] = Resp(429, "slow down", {"Content-Type": "text/plain"})
    for p in ("nous-contacter", "foire-aux-questions", "offre"):
        routes[J + f"https://app.test/{p}"] = Resp(429, "slow down", {"Content-Type": "text/plain"})
    sess = site(routes)
    report = read_site("app.test")
    jina_calls = [c for c in sess.calls if c.startswith(J)]
    assert len(jina_calls) == 2  # the start page, then the first 429 stops it
    assert any("renderer is busy" in w for w in report.warnings) and report.pages[0].rendered


def test_renderer_answer_from_another_host_is_dropped(site):
    routes = js_routes()
    routes[J + "https://app.test/"] = jina("Elsewhere", "https://evil.test/", RATE_MD)
    site(routes)
    report = read_site("app.test")
    assert "https://app.test/" not in [p.url for p in report.pages]


def test_markdown_parsing():
    title, source, md = scraper.parse_jina("Title: Docs¶ \n\nURL Source: https://a.test/x\n\nMarkdown Content:\n# Hi\n\n**Bold** [link text](/y) ![img](a.png)\n")
    assert (title, source) == ("Docs", "https://a.test/x")
    assert scraper.markdown_text(md) == "Hi\n\nBold link text"
    assert scraper.markdown_links(md + "[pdf](/f.pdf) [![i](b.png)](/z)", source) == [("https://a.test/y", "link text"),
                                                                                    ("https://a.test/z", "")]


# ---- crawl order, parallel fetch, crawl-delay, titles ------------------------------------------------------

def test_priority_start_path_then_key_pages_then_the_rest(site):
    links = ["/browse/other", "/browse/driving/licences", "/browse/contact", "/browse/driving/car-tax", "/browse/zzz"]
    routes = {"https://gov.test/robots.txt": Resp(404, b""),
              "https://gov.test/browse/driving": html(page(WORDS, "Driving", links))}
    for i, l in enumerate(links):
        routes["https://gov.test" + l] = html(page(WORDS + f" page {i} unique {l}", l))
    site(routes)
    urls = [p.url for p in read_site("https://gov.test/browse/driving").pages if "#" not in p.url]
    assert urls == ["https://gov.test/browse/driving", "https://gov.test/browse/driving/licences",
                    "https://gov.test/browse/driving/car-tax", "https://gov.test/browse/contact",
                    "https://gov.test/browse/other", "https://gov.test/browse/zzz"]


def test_refinement_pages_wait_behind_new_topics(site):
    links = ["query-params/", "query-params-str-validations/", "body/", "body-fields/", "response-model/",
             "response-status-code/"]
    routes = {"https://d.test/robots.txt": Resp(404, b""), "https://d.test/tutorial/": html(page(WORDS, "Tutorial", links))}
    for l in links:
        routes["https://d.test/tutorial/" + l] = html(page(WORDS + " " + l, l))
    site(routes)
    urls = [p.url.rsplit("/", 2)[-2] for p in read_site("https://d.test/tutorial/", 5).pages]
    assert urls == ["tutorial", "query-params", "body", "response-model", "response-status-code"]


class SlowSession(FakeSession):
    def __init__(self, routes):
        super().__init__(routes)
        self.now = self.max = 0
        self.lock = threading.Lock()

    def get(self, url, **kw):
        with self.lock:
            self.now += 1
            self.max = max(self.max, self.now)
        time.sleep(0.03)
        try:
            return super().get(url, **kw)
        finally:
            with self.lock:
                self.now -= 1


def _many_pages(robots_txt):
    links = [f"/p{i}" for i in range(8)]
    routes = {"https://shop.test/robots.txt": Resp(200, robots_txt, {"Content-Type": "text/plain"}),
              "https://shop.test/": html(page(WORDS, "Home", links))}
    for i, l in enumerate(links):
        routes["https://shop.test" + l] = html(page(WORDS + f" item {i}", f"P{i}"))
    return routes


def test_parallel_fetch_keeps_order_and_counts(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", fake_dns)
    monkeypatch.setattr(scraper, "FETCH_WORKERS", 4)
    sess = SlowSession(_many_pages("User-agent: *\nAllow: /\n"))
    monkeypatch.setattr(scraper.requests, "Session", lambda: sess)
    seen = []
    t0 = time.monotonic()
    report = read_site("shop.test", 9, on_page=lambda n, u: seen.append(n))
    assert len([p for p in report.pages if "#" not in p.url]) == 9 and seen == list(range(1, 10))
    assert [p.url for p in report.pages][:3] == ["https://shop.test/", "https://shop.test/p0", "https://shop.test/p1"]
    assert sess.max > 1 and time.monotonic() - t0 < 1.5


def test_crawl_delay_means_one_request_at_a_time(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", fake_dns)
    monkeypatch.setattr(scraper, "FETCH_WORKERS", 4)
    monkeypatch.setattr(scraper, "MAX_CRAWL_DELAY_S", 0.01)  # honour the mode, not the 2 s wait, in a unit test
    monkeypatch.setattr(scraper, "DELAY_S", 0)
    sess = SlowSession(_many_pages("User-agent: *\nCrawl-delay: 2\n"))
    monkeypatch.setattr(scraper.requests, "Session", lambda: sess)
    report = read_site("shop.test", 9)
    assert sess.max == 1 and len([p for p in report.pages if "#" not in p.url]) == 9


def test_titles_lose_pilcrows_and_the_repeated_brand_suffix():
    pages = [scraper.Page(f"https://a.test/{i}", t, "x") for i, t in enumerate(
        ["Quickstart¶ | Acme Docs", "Install  | Acme Docs", "Acme Docs", "API | Acme Docs", "FAQ - Other"])]
    scraper.clean_titles(pages)
    assert [p.title for p in pages] == ["Quickstart", "Install", "Acme Docs", "API", "FAQ - Other"]


def test_site_info_page_keeps_footer_and_phone_links(site):
    footer = '<footer><p>Call us 9am to 5pm</p><a href="tel:+212522000000">Phone</a><a href="mailto:hi@shop.test">Mail</a></footer>'
    routes = {"https://shop.test/robots.txt": Resp(404, b""),
              "https://shop.test/": html(page(WORDS, "Home", ["/faq"]).replace("</body>", footer + "</body>")),
              "https://shop.test/faq": html(page(WORDS + " faq", "FAQ").replace("</body>", footer + "</body>"))}
    site(routes)
    pages = read_site("shop.test").pages
    info = pages[-1]
    assert info.title == scraper.SITE_INFO_TITLE and info.url == "https://shop.test/#site-info"
    assert "Phone: +212522000000" in info.text and "Email: hi@shop.test" in info.text
    assert info.text.count("9am to 5pm") == 1
