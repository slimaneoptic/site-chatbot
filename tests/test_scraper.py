"""Crawler tests: no network. A fake requests session serves HTML strings; DNS is faked too."""
import io
import ipaddress
import socket
import sys
from pathlib import Path

import pytest
import requests
from requests.structures import CaseInsensitiveDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scraper  # noqa: E402
from scraper import BlockedURL, CrawlError, check_public_url, crawl, fetch, looks_js_only, normalize_url, read_site  # noqa: E402

WORDS = "Our shop ships orders to Morocco, Saudi Arabia and the UAE within three to five working days. " * 4
CRA = '<html><head><script src="/static/js/main.js"></script></head><body><noscript>You need to enable JavaScript to run this app.</noscript><div id="root"></div></body></html>'


def page(body: str, title: str = "Page", links=()) -> str:
    a = "".join(f'<a href="{href}">link</a> ' for href in links)
    return f"<html><head><title>{title}</title></head><body><main><h1>{title}</h1><p>{body}</p></main><nav>{a}</nav></body></html>"


class Resp:
    def __init__(self, status=200, body=b"", headers=None, encoding="utf-8"):
        self.status_code = status
        self._body = body.encode() if isinstance(body, str) else body
        self.headers = CaseInsensitiveDict(headers if headers is not None else {"Content-Type": "text/html; charset=utf-8"})
        self.encoding = encoding

    def iter_content(self, n):
        for i in range(0, len(self._body), n):
            yield self._body[i:i + n]

    def close(self):
        pass


def html(body, **kw):
    return Resp(200, body, **kw)


def redirect(to, status=301):
    return Resp(status, b"", {"Location": to})


class FakeSession:
    def __init__(self, routes):
        self.routes, self.headers, self.calls = routes, {}, []

    def get(self, url, **kw):
        self.calls.append(url)
        got = self.routes.get(url, Resp(404, b"", {"Content-Type": "text/html"}))
        if isinstance(got, Exception):
            raise got
        return got() if callable(got) else got

    def close(self):
        pass


PRIVATE_HOSTS = {"intranet.test": "10.0.0.5", "mapped.test": "::ffff:127.0.0.1", "nat64.test": "64:ff9b::7f00:1", "meta.test": "169.254.169.254"}


def fake_dns(host, *a, **kw):
    if host == "nowhere.test":
        raise socket.gaierror("no such host")
    try:
        ip = str(ipaddress.ip_address(host))  # literal addresses resolve to themselves, as in real DNS
    except ValueError:
        ip = PRIVATE_HOSTS.get(host, "93.184.216.34")
    fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
    return [(fam, socket.SOCK_STREAM, 6, "", (ip, 0))]


@pytest.fixture
def site(monkeypatch):
    """Install a fake site: site(routes) -> FakeSession."""
    monkeypatch.setattr(socket, "getaddrinfo", fake_dns)
    monkeypatch.setattr(scraper, "DELAY_S", 0)
    monkeypatch.setattr(scraper, "FETCH_WORKERS", 1)  # deterministic order; parallel reads have their own tests
    monkeypatch.setattr(scraper, "_JINA_BUCKET", scraper._Bucket())

    def install(routes):
        sess = FakeSession(routes)
        monkeypatch.setattr(scraper.requests, "Session", lambda: sess)
        return sess
    return install


# ---- addresses ----------------------------------------------------------------------------------

@pytest.mark.parametrize("raw, want", [
    ("example.com/help", "https://example.com/help"),
    ("  HTTP://Example.COM  ", "http://example.com/"),
    ("https://example.com:443/docs/#top", "https://example.com/docs/"),
    ("//example.com/a?x=1", "https://example.com/a?x=1"),
    ("example.com:8080", "https://example.com:8080/"),
])
def test_normalize_url_accepts_missing_scheme(raw, want):
    assert normalize_url(raw) == want


@pytest.mark.parametrize("raw", ["", "   ", "exa mple.com", "ftp://example.com/", "http://example.com:abc/", "javascript:alert(1)",
                                 "http://user:pw@example.com/", "http://[::1", "https://", None])
def test_normalize_url_rejects_with_friendly_crawl_error(raw):
    with pytest.raises(CrawlError) as info:
        normalize_url(raw)
    assert type(info.value) is CrawlError and info.value.code == "bad_url"
    assert info.value.message and "Traceback" not in str(info.value)


def test_blocked_url_is_still_a_value_error_and_old_style_works():
    err = BlockedURL("custom text")
    assert isinstance(err, ValueError) and isinstance(err, CrawlError)
    assert err.code == "blocked" and str(err) == "custom text"
    assert BlockedURL("bad_url", "x").code == "bad_url"


@pytest.mark.parametrize("url, code", [
    ("http://intranet.test/", "blocked"), ("http://mapped.test/", "blocked"), ("http://nat64.test/", "blocked"),
    ("http://meta.test/latest/", "blocked"), ("http://[::1]/", "blocked"), ("http://nowhere.test/", "bad_url"),
    ("http://example.com:abc/", "bad_url"), ("http://example.com:8443/", "bad_url"), ("file:///etc/passwd", "bad_url"),
])
def test_check_public_url_codes(monkeypatch, url, code):
    monkeypatch.setattr(socket, "getaddrinfo", fake_dns)
    with pytest.raises(BlockedURL) as info:
        check_public_url(url)
    assert info.value.code == code


def test_check_public_url_allows_public_host(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", fake_dns)
    check_public_url("https://shop.test/help/")


# ---- crawling ---------------------------------------------------------------------------------------

def shop_routes():
    return {
        "https://shop.test/robots.txt": Resp(200, "User-agent: *\nDisallow: /help/private/\n", {"Content-Type": "text/plain"}),
        "https://shop.test/help/": html(page(WORDS, "Help", ["shipping.html", "/help/returns.html", "/help/private/x.html",
                                                             "/about.html", "https://other.test/help/", "file.pdf", "#top"])),
        "https://shop.test/help/shipping.html": html(page(WORDS, "Shipping", ["/help/"])),
        "https://shop.test/help/returns.html": html(page("Returns are accepted within 30 days for a full refund. " * 6, "Returns")),
    }


def test_read_site_stays_in_folder_respects_robots_and_reports(site):
    sess = site(shop_routes())
    seen = []
    report = read_site("shop.test/help/", 10, on_page=lambda n, u: seen.append((n, u)))
    assert report.start_url == "https://shop.test/help/"
    assert [p.url for p in report.pages] == ["https://shop.test/help/", "https://shop.test/help/shipping.html", "https://shop.test/help/returns.html"]
    assert seen[-1] == (3, "https://shop.test/help/returns.html")
    # /about.html is outside the folder but a key page linked from the start page: one hop is allowed (404 here)
    assert report.stopped == "done" and report.js_only is False and report.fetched == 4
    assert "https://shop.test/about.html" in sess.calls
    assert not any("private" in c or "other.test" in c or c.endswith(".pdf") or "jina" in c for c in sess.calls)
    assert any("robots.txt" in w for w in report.warnings)


def test_old_crawl_api_still_yields_pages_and_accepts_no_scheme(site):
    site(shop_routes())
    pages = list(crawl("shop.test/help/", 2))
    assert len(pages) == 2 and all(isinstance(p, scraper.Page) for p in pages)


@pytest.mark.parametrize("route, code", [
    (Resp(404, b"", {"Content-Type": "text/html"}), "not_found"),
    (requests.ConnectionError("refused"), "unreachable"),
    (requests.ReadTimeout("slow"), "timeout"),
    (Resp(503, b"", {"Content-Type": "text/html"}), "unreachable"),
    (Resp(200, b"%PDF-1.4", {"Content-Type": "application/pdf"}), "no_text"),
    (redirect("http://intranet.test/admin"), "blocked"),
])
def test_start_page_failures_are_typed(site, route, code):
    site({"https://shop.test/robots.txt": Resp(404, b""), "https://shop.test/help/": route})
    with pytest.raises(CrawlError) as info:
        read_site("https://shop.test/help/")
    assert info.value.code == code and info.value.message


def test_old_crawl_yields_nothing_when_start_page_fails_but_raises_blocked_for_bad_urls(site):
    site({"https://shop.test/help/": Resp(404, b"")})
    assert list(crawl("https://shop.test/help/")) == []
    with pytest.raises(BlockedURL):
        list(crawl("exa mple.com"))
    with pytest.raises(BlockedURL):
        list(crawl("http://intranet.test/"))


def test_robots_blocks_start_page_and_server_error_robots_means_stay_out(site):
    site({"https://shop.test/robots.txt": Resp(200, "User-agent: *\nDisallow: /\n", {"Content-Type": "text/plain"})})
    with pytest.raises(CrawlError) as info:
        read_site("https://shop.test/help/")
    assert info.value.code == "robots"
    sess = site({"https://shop.test/robots.txt": Resp(500, b""), "https://shop.test/help/": html(page(WORDS))})
    with pytest.raises(CrawlError) as info:
        read_site("https://shop.test/help/")
    assert info.value.code == "robots" and "https://shop.test/help/" not in sess.calls


def test_robots_txt_is_capped(site, monkeypatch):
    monkeypatch.setattr(scraper, "ROBOTS_MAX_BYTES", 200)
    routes = shop_routes()
    routes["https://shop.test/robots.txt"] = Resp(200, "User-agent: *\nDisallow: /help/returns\n" + "# filler\n" * 50_000, {"Content-Type": "text/plain"})
    site(routes)
    report = read_site("https://shop.test/help/")
    assert "https://shop.test/help/returns.html" not in [p.url for p in report.pages]


def test_redirect_to_internal_address_mid_crawl_is_skipped(site):
    routes = shop_routes()
    routes["https://shop.test/help/shipping.html"] = redirect("http://meta.test/latest/meta-data/")
    sess = site(routes)
    report = read_site("https://shop.test/help/")
    assert [p.url for p in report.pages] == ["https://shop.test/help/", "https://shop.test/help/returns.html"]
    assert not any("meta.test" in c for c in sess.calls)


def test_start_redirect_to_www_follows_the_new_host(site):
    site({
        "https://shop.test/": redirect("https://www.shop.test/"),
        "https://www.shop.test/": html(page(WORDS, "Home", ["/faq"])),
        "https://www.shop.test/faq": html(page("Questions and answers about orders, payment and delivery. " * 6, "FAQ")),
    })
    report = read_site("shop.test")
    assert [p.url for p in report.pages] == ["https://www.shop.test/", "https://www.shop.test/faq"]


def test_thin_pages_count_toward_fetch_cap(site):
    routes = {"https://shop.test/robots.txt": Resp(404, b"")}
    for i in range(50):
        routes[f"https://shop.test/p{i}"] = html(page("tiny", f"P{i}", [f"/p{j}" for j in range(i + 1, i + 4)]))
    sess = site(routes)
    report = read_site("https://shop.test/p0", 2)
    assert report.pages == [] and report.stopped == "max_fetches"
    assert report.fetched == 6 and len([c for c in sess.calls if not any(x in c for x in ("robots", "jina", "sitemap"))]) == 6
    assert report.warnings


def test_time_limit_stops_the_crawl(site, monkeypatch):
    class Clock:
        now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, s):
            self.now += s

    clock = Clock()
    monkeypatch.setattr(scraper, "time", clock)
    routes = shop_routes()

    def slow(resp):
        def go():
            clock.now += 25
            return resp
        return go
    for k in list(routes):
        if not k.endswith("robots.txt"):
            routes[k] = slow(routes[k])
    site(routes)
    report = read_site("https://shop.test/help/", 10, max_seconds=60)
    assert report.stopped == "time_limit" and len(report.pages) == 2
    assert any("60 seconds" in w for w in report.warnings)


def test_js_only_site_is_reported_not_raised(site):
    site({"https://app.test/robots.txt": Resp(404, b""), "https://app.test/": html(CRA)})
    report = read_site("app.test")
    assert report.pages == [] and report.js_only is True
    assert "JavaScript" in report.warnings[0]


def test_empty_plain_site_gets_no_text_warning(site):
    site({"https://shop.test/robots.txt": Resp(404, b""), "https://shop.test/": html(page("Hello"))})
    report = read_site("shop.test")
    assert report.pages == [] and report.js_only is False and "No readable text" in report.warnings[0]


def test_looks_js_only():
    assert looks_js_only(CRA, "")
    assert looks_js_only('<div id="__next"></div><script id="__NEXT_DATA__">{}</script>', "")
    assert looks_js_only("<body>" + "<script src=a.js></script>" * 4 + "<div></div></body>", "")
    assert not looks_js_only(page(WORDS), WORDS)
    assert not looks_js_only(page("Contact us by email or phone, we answer every day from nine to five, Monday to Friday."), "")


def test_unknown_charset_and_old_fetch_behaviour(site):
    sess = FakeSession({
        "https://shop.test/a": Resp(200, "café".encode(), {"Content-Type": "text/html; charset=x-made-up"}, encoding="x-made-up"),
        "https://shop.test/missing": Resp(404, b""),
        "https://shop.test/img": Resp(200, b"\x89PNG", {"Content-Type": "image/png"}),
        "https://shop.test/down": requests.ConnectionError("down"),
    })
    site({})
    assert fetch(sess, "https://shop.test/a") == ("https://shop.test/a", "café")
    assert fetch(sess, "https://shop.test/missing") is None
    assert fetch(sess, "https://shop.test/img") is None
    with pytest.raises(requests.ConnectionError):
        fetch(sess, "https://shop.test/down")
    with pytest.raises(BlockedURL):
        fetch(sess, "http://intranet.test/")


def test_big_page_is_skipped(site, monkeypatch):
    monkeypatch.setattr(scraper, "MAX_BYTES", 1000)
    routes = shop_routes()
    routes["https://shop.test/help/shipping.html"] = html(page(WORDS * 20, "Huge"))
    site(routes)
    assert "https://shop.test/help/shipping.html" not in [p.url for p in read_site("https://shop.test/help/").pages]


def test_unexpected_bug_becomes_friendly_crawl_error(site, monkeypatch):
    site(shop_routes())
    monkeypatch.setattr(scraper, "_links", lambda *a: 1 / 0)
    with pytest.raises(CrawlError) as info:
        read_site("https://shop.test/help/")
    assert "ZeroDivision" not in info.value.message and info.value.code == "unreachable"


def test_extractor_crash_counts_as_thin_page(site, monkeypatch):
    site(shop_routes())

    def boom(*a, **kw):
        raise RuntimeError("parser exploded")
    monkeypatch.setattr(scraper.trafilatura, "bare_extraction", boom)
    report = read_site("https://shop.test/help/")
    assert report.pages == [] and report.warnings


def test_callers_own_error_in_on_page_passes_through(site):
    site(shop_routes())

    class Mine(Exception):
        pass

    def on_page(n, u):
        raise Mine()
    with pytest.raises(Mine):
        read_site("https://shop.test/help/", on_page=on_page)


def test_duplicate_pages_kept_once(site):
    routes = shop_routes()
    routes["https://shop.test/help/shipping.html"] = redirect("https://shop.test/help/")
    site(routes)
    urls = [p.url for p in read_site("https://shop.test/help/").pages]
    assert urls.count("https://shop.test/help/") == 1


# ---- DNS rebinding: the address actually connected to is checked (fake sockets, no network) -----------

class FakeSock:
    """Just enough of a socket for urllib3/http.client: records what was sent, answers with `reply`."""

    def __init__(self, peer, reply=b""):
        self.peer, self.reply, self.sent, self.closed = peer, reply, b"", False

    def getpeername(self):
        return (self.peer, 80)

    def sendall(self, data):
        self.sent += bytes(data)

    def makefile(self, *a, **kw):
        return io.BytesIO(self.reply)

    def settimeout(self, t):
        pass

    def gettimeout(self):
        return None

    def setsockopt(self, *a):
        pass

    def close(self):
        self.closed = True


def http_reply(body=b"", status="200 OK", headers=()):
    head = [f"HTTP/1.1 {status}", f"Content-Length: {len(body)}", "Connection: close", *headers]
    return ("\r\n".join(head) + "\r\n\r\n").encode() + body


@pytest.fixture
def rebinding(monkeypatch):
    """DNS check says public (fake_dns), but the connect lands on whatever `peers[host]` says."""
    import urllib3.util.connection as u3conn
    monkeypatch.setattr(socket, "getaddrinfo", fake_dns)
    peers, socks = {}, []

    def connect(address, *a, **kw):
        peer, reply = peers[address[0]]
        socks.append(FakeSock(peer, reply))
        return socks[-1]
    monkeypatch.setattr(u3conn, "create_connection", connect)
    return peers, socks


def test_rebinding_host_is_blocked_at_connect(rebinding):
    peers, socks = rebinding
    peers["rebind.test"] = ("127.0.0.1", http_reply(b"secret"))
    with requests.Session() as sess, pytest.raises(BlockedURL) as err:
        scraper._download(sess, "http://rebind.test/")
    assert err.value.code == "blocked"
    assert socks[0].sent == b"" and socks[0].closed  # not one byte went to the private address


def test_public_peer_still_works_and_proxies_are_ignored(rebinding, monkeypatch):
    peers, socks = rebinding
    monkeypatch.setenv("HTTP_PROXY", "http://10.0.0.9:3128")
    peers["shop.test"] = ("93.184.216.34", http_reply(b"<p>hi</p>", headers=["Content-Type: text/html"]))
    with requests.Session() as sess:
        final, status, headers, body, _ = scraper._download(sess, "http://shop.test/a")
        assert isinstance(sess.get_adapter("https://shop.test/"), scraper.PublicOnlyAdapter)
    assert (final, status, body) == ("http://shop.test/a", 200, b"<p>hi</p>")
    assert socks[0].sent.startswith(b"GET /a HTTP/1.1")


def test_redirect_hop_to_rebinding_host_is_blocked(rebinding):
    peers, socks = rebinding
    peers["shop.test"] = ("93.184.216.34", http_reply(status="302 Found", headers=["Location: http://other.test/x"]))
    peers["other.test"] = ("10.0.0.5", http_reply(b"internal"))
    with requests.Session() as sess, pytest.raises(BlockedURL):
        scraper._download(sess, "http://shop.test/")
    assert [s.peer for s in socks] == ["93.184.216.34", "10.0.0.5"] and socks[1].sent == b""


@pytest.mark.parametrize("peer", ["169.254.169.254", "::ffff:127.0.0.1", "64:ff9b::a00:5", "::1", ""])
def test_https_connection_checks_peer_before_tls(rebinding, peer):
    peers, socks = rebinding
    peers["rebind.test"] = (peer, b"")
    conn = scraper._PublicOnlyHTTPSConnection("rebind.test", 443)
    with pytest.raises(BlockedURL):
        conn._new_conn()
    assert socks[0].closed
    pools = scraper.PublicOnlyAdapter().poolmanager.pool_classes_by_scheme
    assert pools["https"].ConnectionCls is scraper._PublicOnlyHTTPSConnection
    assert pools["http"].ConnectionCls is scraper._PublicOnlyHTTPConnection


def test_read_site_start_page_on_rebinding_host_is_blocked(rebinding, monkeypatch):
    peers, socks = rebinding
    monkeypatch.setattr(scraper, "DELAY_S", 0)
    peers["rebind.test"] = ("192.168.1.1", http_reply(b"router"))
    with pytest.raises(BlockedURL):
        read_site("http://rebind.test/")
    assert all(s.sent == b"" for s in socks)


@pytest.mark.parametrize("raw", ["::7f00:1", "::a00:1", "::c0a8:1", "::8.8.8.8", "100::1"])
def test_ipv4_compatible_and_reserved_ipv6_are_not_public(raw):
    assert scraper._public_ip(raw) is False


def test_public_ipv6_still_public():
    assert scraper._public_ip("2606:4700:4700::1111") is True and scraper._public_ip("8.8.8.8") is True


def test_terminal_colour_markup_is_stripped_but_code_is_kept():
    raw = ('$ <font color="#4E9A06">uv run fastapi</font> dev\n  <span style="background-color:#009485">'
           '<font color="#D3D7CF"> FastAPI </font></span> <u style="text-decoration-style:solid"><b>main</b></u>')
    assert scraper.strip_inline_html(raw) == "$ uv run fastapi dev\n   FastAPI  main"
    code = "if a < b and c > d: x = List<int>() # <module> <stdin>"
    assert scraper.strip_inline_html(code) == code
