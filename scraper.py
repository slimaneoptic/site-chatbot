"""Polite same-site crawler: robots.txt, rate limit, size and time caps, no internal addresses.

Every failure a visitor can cause comes out as a CrawlError with a short, friendly message.
DNS rebinding: check_public_url() resolves the host, but the HTTP library resolves it again when it connects,
so a host could answer "public" first and "private" next. Every session used here therefore gets
PublicOnlyAdapter: right after each TCP connect (every hop, every redirect, before TLS or any byte is sent)
the address actually connected to is checked with _public_ip(), and a private one raises BlockedURL.
"""
from __future__ import annotations

import hashlib
import ipaddress
import logging
import re
import socket
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterator
from urllib import robotparser
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse

import lxml.html
import requests
import trafilatura
from requests.adapters import HTTPAdapter
from requests.sessions import Session as _Session  # the class itself, even if a test swaps requests.Session
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool

USER_AGENT = "SiteChatbotDemo/1.0 (portfolio demo; polite crawler)"
MAX_BYTES = 2_000_000
TIMEOUT_S = 10
DELAY_S = 0.5
SKIP_EXT = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".zip", ".gz", ".mp4", ".mp3", ".css", ".js", ".xml", ".ico", ".woff", ".woff2")

MAX_FETCHES_FACTOR = 3      # fetch attempts <= max_pages * 3 (thin pages count too)
MAX_SECONDS = 60            # whole crawl
ROBOTS_MAX_BYTES = 500_000  # longer robots.txt files are cut here (RFC 9309 asks for at least 500 KiB)
MIN_WORDS = 30              # a page with fewer words is "thin" and is not kept
MAX_CRAWL_DELAY_S = 5       # robots.txt Crawl-delay is respected up to this
REDIRECTS = (301, 302, 303, 307, 308)

log = logging.getLogger(__name__)

MESSAGES = {
    "bad_url": "That doesn't look like a website address. Try something like example.com/help/.",
    "blocked": "That address points to a private or internal network, so it can't be read.",
    "not_found": "That page doesn't exist (the site answered \"not found\"). Check the address.",
    "unreachable": "Couldn't reach that website. Check the address or try again in a minute.",
    "timeout": "The website took too long to answer. Try again, or try a smaller section of the site.",
    "robots": "This site's robots.txt asks crawlers not to read that address, so it is skipped.",
    "no_text": "No readable text found there. Try a more specific address, such as a help or docs section.",
    "js_only": "This site draws its text with JavaScript, so a simple reader sees an empty page. "
               "Client versions use a real browser for sites like this.",
}


@dataclass
class Page:
    url: str
    title: str
    text: str

    @property
    def words(self) -> int:
        return len(self.text.split())


class CrawlError(Exception):
    """A crawl problem the visitor can understand. `code` is for code, `message` is for people."""

    def __init__(self, code: str, message: str | None = None):
        self.code = code
        self.message = message or MESSAGES.get(code, MESSAGES["unreachable"])
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class BlockedURL(CrawlError, ValueError):
    """Unsafe or unusable address. Still a ValueError; BlockedURL("text") keeps working (code "blocked")."""

    def __init__(self, code_or_message: str, message: str | None = None):
        if message is None and code_or_message not in MESSAGES:
            code, message = "blocked", code_or_message
        else:
            code = code_or_message
        super().__init__(code, message)


@dataclass
class CrawlReport:
    start_url: str
    pages: list[Page]
    fetched: int
    stopped: str            # "done" | "max_pages" | "max_fetches" | "time_limit"
    js_only: bool
    warnings: list[str] = field(default_factory=list)


def normalize_url(url: str) -> str:
    """Clean a pasted address: add https:// when missing, lower-case the host, drop default ports and #fragments."""
    url = url.strip() if isinstance(url, str) else ""
    if not url:
        raise CrawlError("bad_url", "Paste a website address, for example example.com/help/.")
    if len(url) > 2000 or any(c.isspace() for c in url):
        raise CrawlError("bad_url", "A website address can't contain spaces.")
    if url.startswith("//"):
        url = "https:" + url
    elif "://" not in url:
        url = "https://" + url
    try:
        parts = urlparse(url)
        port = parts.port
    except ValueError:
        raise CrawlError("bad_url", "That address has an invalid host or port number.") from None
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise CrawlError("bad_url", "Only http:// and https:// website addresses are supported.")
    if "@" in parts.netloc:
        raise CrawlError("bad_url", "Addresses with a login in them (user@site) are not supported.")
    host = parts.hostname
    if not host or host.startswith(".") or ".." in host:
        raise CrawlError("bad_url", "That address has no website name in it.")
    if ":" in host:
        host = f"[{host}]"
    if port is not None and port != {"http": 80, "https": 443}[scheme]:
        host = f"{host}:{port}"
    return urlunparse((scheme, host, parts.path or "/", parts.params, parts.query, ""))


_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_V4_COMPAT = ipaddress.ip_network("::/96")  # deprecated IPv4-compatible form ::a.b.c.d


def _public_ip(raw: str) -> bool:
    try:
        ip = ipaddress.ip_address(raw.split("%", 1)[0])
    except ValueError:
        return False
    if ip.version == 6:  # IPv6 forms that carry an IPv4 address inside
        inner = ip.ipv4_mapped or ip.sixtofour or (ip.teredo[1] if ip.teredo else None)
        if inner is None and (ip in _NAT64 or ip in _V4_COMPAT):
            inner = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        if inner is not None and not _public_ip(str(inner)):
            return False
    return ip.is_global and not ip.is_multicast and not ip.is_reserved


def check_public_url(url: str) -> None:
    """Reject anything that is not plain http(s) to a public internet address."""
    try:
        parts = urlparse(url)
        port = parts.port
        host = parts.hostname
    except ValueError:
        raise BlockedURL("bad_url", "That address has an invalid host or port number.") from None
    if parts.scheme not in ("http", "https") or not host:
        raise BlockedURL("bad_url", "Only http:// and https:// website addresses are supported.")
    if port not in (None, 80, 443):
        raise BlockedURL("bad_url", "Only standard web ports (80/443) are allowed.")
    try:
        infos = socket.getaddrinfo(host, None)
    except (OSError, UnicodeError):
        raise BlockedURL("bad_url", f"Can't find the website {host[:100]}. Check the address for typos.") from None
    if not infos:
        raise BlockedURL("bad_url", f"Can't find the website {host[:100]}. Check the address for typos.")
    for info in infos:
        if not _public_ip(str(info[4][0])):
            raise BlockedURL("blocked", "That address points to a private or internal network.")


def _check_peer(sock) -> None:
    """Raise BlockedURL (and close the socket) unless the socket is connected to a public address."""
    try:
        peer = str(sock.getpeername()[0])
    except (OSError, IndexError, TypeError, AttributeError):
        peer = ""
    if not _public_ip(peer):
        try:
            sock.close()
        except OSError:
            pass
        raise BlockedURL("blocked", "That address points to a private or internal network.")


class _PublicOnlyHTTPConnection(HTTPConnection):
    def _new_conn(self):
        sock = super()._new_conn()
        _check_peer(sock)
        return sock


class _PublicOnlyHTTPSConnection(HTTPSConnection):
    def _new_conn(self):  # runs before the TLS handshake
        sock = super()._new_conn()
        _check_peer(sock)
        return sock


class _PublicOnlyHTTPPool(HTTPConnectionPool):
    ConnectionCls = _PublicOnlyHTTPConnection


class _PublicOnlyHTTPSPool(HTTPSConnectionPool):
    ConnectionCls = _PublicOnlyHTTPSConnection


class PublicOnlyAdapter(HTTPAdapter):
    """requests adapter whose connections check the connected peer address (closes the DNS rebinding gap)."""

    def init_poolmanager(self, *args, **kwargs):
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = {"http": _PublicOnlyHTTPPool, "https": _PublicOnlyHTTPSPool}

    def proxy_manager_for(self, proxy, **proxy_kwargs):  # a proxy would hide the real peer: never use one
        raise BlockedURL("unreachable", "The demo reader does not use proxies.")


def _harden(session) -> None:
    """Mount PublicOnlyAdapter on a real requests.Session (once) and ignore proxy env vars. Test fakes pass through."""
    if isinstance(session, _Session) and not isinstance(session.get_adapter("https://x.invalid/"), PublicOnlyAdapter):
        session.trust_env = False
        session.proxies = {}
        session.mount("http://", PublicOnlyAdapter())
        session.mount("https://", PublicOnlyAdapter())


def _download(session, url: str, *, max_redirects: int = 3, max_bytes: int | None = None, truncate: bool = False,
              deadline: float | None = None, allow: Callable[[str], bool] | None = None):
    """GET with every redirect hop re-checked. Returns (final_url, status, headers, body, encoding); raises CrawlError."""
    max_bytes = MAX_BYTES if max_bytes is None else max_bytes
    _harden(session)
    stop_at = time.monotonic() + 3 * TIMEOUT_S
    if deadline is not None:
        stop_at = min(stop_at, deadline)
    for _ in range(max_redirects + 1):
        check_public_url(url)
        if allow is not None and not allow(url):
            raise CrawlError("robots", "Skipped: outside the allowed pages.")
        left = stop_at - time.monotonic()
        if left <= 0:
            raise CrawlError("timeout")
        try:
            resp = session.get(url, timeout=min(TIMEOUT_S, max(left, 0.5)), allow_redirects=False, stream=True)
        except requests.Timeout as err:
            raise CrawlError("timeout") from err
        except requests.RequestException as err:
            raise CrawlError("unreachable") from err
        try:
            if resp.status_code in REDIRECTS and resp.headers.get("Location"):
                url = urljoin(url, resp.headers["Location"])
                continue
            body = b""
            if resp.status_code == 200:
                for chunk in resp.iter_content(64_000):
                    body += chunk
                    if len(body) > max_bytes:
                        if truncate:
                            body = body[:max_bytes]
                            break
                        raise CrawlError("no_text", f"That page is too large for the demo (over {max_bytes // 1_000_000 or 1} MB).")
                    if time.monotonic() > stop_at:
                        raise CrawlError("timeout")
            return url, resp.status_code, resp.headers, body, resp.encoding
        except requests.Timeout as err:
            raise CrawlError("timeout") from err
        except requests.RequestException as err:
            raise CrawlError("unreachable") from err
        finally:
            resp.close()
    raise CrawlError("unreachable", "The site redirected too many times.")


def _decode(body: bytes, charset: str | None) -> str:
    try:
        return body.decode(charset or "utf-8", errors="replace")
    except LookupError:  # unknown charset name in the header
        return body.decode("utf-8", errors="replace")


def _fetch_html(session, url: str, max_redirects: int = 3, *, deadline: float | None = None,
                allow: Callable[[str], bool] | None = None) -> tuple[str, str]:
    final, status, headers, body, encoding = _download(session, url, max_redirects=max_redirects, deadline=deadline, allow=allow)
    if status in (404, 410):
        raise CrawlError("not_found")
    if status in (401, 403):
        raise CrawlError("unreachable", f"The site refused the request (HTTP {status}). Some sites block automated readers.")
    if status == 429:
        raise CrawlError("unreachable", "The site is limiting requests right now (HTTP 429). Try again in a few minutes.")
    if status != 200:
        raise CrawlError("unreachable", f"The site answered with an error (HTTP {status}). Try again later.")
    ctype = (headers.get("Content-Type") or "").lower()
    if "text/html" not in ctype and "application/xhtml" not in ctype:
        kind = ctype.split(";")[0].strip() or "unknown type"
        hint = " For PDFs, use the PDF upload instead." if "pdf" in kind else ""
        raise CrawlError("no_text", f"That address is not a web page ({kind[:60]}).{hint}")
    return final, _decode(body, encoding if "charset" in ctype else None)


def fetch(session: requests.Session, url: str, max_redirects: int = 3) -> tuple[str, str] | None:
    """Return (final_url, html) or None. Every redirect hop is re-checked.
    Raises BlockedURL for unsafe hops and requests errors for network failures, as before."""
    try:
        return _fetch_html(session, url, max_redirects)
    except BlockedURL:
        raise
    except CrawlError as err:
        if isinstance(err.__cause__, requests.RequestException):
            raise err.__cause__ from None
        return None


def _scope(start_url: str) -> tuple[str, str]:
    p = urlparse(start_url)
    path = p.path if p.path.endswith("/") else p.path.rsplit("/", 1)[0] + "/"
    return p.netloc, path


def _links(html: str, base: str) -> list[str]:
    try:
        doc = lxml.html.fromstring(html)
    except (ValueError, lxml.etree.ParserError):
        return []
    doc.make_links_absolute(base)
    out = []
    for el, attr, link, _ in doc.iterlinks():
        if el.tag == "a" and attr == "href":
            link = urldefrag(link)[0]
            if not link.lower().split("?")[0].endswith(SKIP_EXT):
                out.append(link)
    return out


_JS_MARKERS = re.compile(
    r"""id=["'](?:root|app|__next|__nuxt|___gatsby|svelte)["']|__NEXT_DATA__|window\.__NUXT__|ng-version=|data-reactroot"""
    r"""|<app-root|data-server-rendered|enable javascript|javascript is (?:disabled|required)|requires javascript"""
    r"""|turn on javascript|needs javascript""",
    re.I,
)


def _visible_words(html: str) -> int:
    try:
        doc = lxml.html.fromstring(html)
        for el in doc.xpath("//script|//style|//noscript|//template"):
            el.drop_tree()
        return len(doc.text_content().split())
    except Exception:  # noqa: BLE001 - any parser trouble means "no visible text"
        return 0


def looks_js_only(html: str, text: str) -> bool:
    """True when HTML came back with almost no text and looks like a JavaScript app shell."""
    if len((text or "").split()) >= MIN_WORDS:
        return False
    html = html or ""
    if _JS_MARKERS.search(html):
        return True
    return len(re.findall(r"<script\b", html, re.I)) >= 3 and _visible_words(html) < 10


def _extract(html: str, url: str) -> tuple[str, str]:
    """(title, text) of the main content; ("", "") when extraction fails."""
    try:
        data = trafilatura.bare_extraction(html, url=url, with_metadata=True, include_comments=False) or {}
        if not isinstance(data, dict):  # trafilatura 2 returns a Document
            data = data.as_dict()
    except Exception:  # noqa: BLE001 - a broken page must not stop the crawl
        log.warning("text extraction failed for %s", url, exc_info=True)
        return "", ""
    text = strip_inline_html((data.get("text") or "").replace("¶", "")).strip()
    return (data.get("title") or "").strip(), text


# Colour markup that docs keep as text inside terminal blocks (FastAPI: <font color="#4E9A06">fastapi</font>).
# Only these presentational tags, with plain attributes: code such as `a < b` or `List<int>` is left alone.
_INLINE_TAG = re.compile(
    r"</?(?:font|span|u|b|i|em|strong|mark|small|kbd)(?:\s+[a-zA-Z-]+=(?:\"[^\"<>]*\"|'[^'<>]*'))*\s*/?>",
    re.IGNORECASE)


def strip_inline_html(text: str) -> str:
    """Drop inline colour/format tags left in extracted text; the words inside them stay."""
    return _INLINE_TAG.sub("", text or "")


def _load_robots(session, scheme: str, host: str, deadline: float) -> robotparser.RobotFileParser:
    """RFC 9309: 4xx = no rules, 5xx = stay out, network trouble = the site is unreachable."""
    robots = robotparser.RobotFileParser()
    try:
        _, status, _, body, _ = _download(session, f"{scheme}://{host}/robots.txt", max_redirects=4,
                                          max_bytes=ROBOTS_MAX_BYTES, truncate=True, deadline=deadline)
    except CrawlError as err:
        if err.code in ("timeout", "unreachable") and isinstance(err.__cause__, requests.RequestException):
            raise
        robots.parse([])  # e.g. robots.txt redirects somewhere odd: treat as "no rules"
        return robots
    if status == 200:
        lines = _decode(body, "utf-8").splitlines()
        if len(body) >= ROBOTS_MAX_BYTES and lines:
            lines = lines[:-1]  # the last line may be cut in half
        robots.parse(lines)
    elif status >= 500:
        robots.parse(["User-agent: *", "Disallow: /"])
    else:
        robots.parse([])
    return robots


class _Run:
    def __init__(self):
        self.start_url = ""
        self.fetched = 0
        self.stopped = "done"
        self.start_js = False
        self.js_pages = 0
        self.robots_skipped = 0


def _crawl(start_url: str, max_pages: int, on_page, max_seconds: float, run: _Run) -> Iterator[Page]:
    start_url = normalize_url(start_url)
    run.start_url = start_url
    check_public_url(start_url)
    if max_pages <= 0:
        run.stopped = "max_pages"
        return
    deadline = time.monotonic() + max_seconds
    max_fetches = max_pages * MAX_FETCHES_FACTOR
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    try:
        scheme = urlparse(start_url).scheme
        host, prefix = _scope(start_url)
        robots = _load_robots(session, scheme, host, deadline)
        if not robots.can_fetch(USER_AGENT, start_url):
            raise CrawlError("robots")
        delay = max(DELAY_S, min(float(robots.crawl_delay(USER_AGENT) or 0), MAX_CRAWL_DELAY_S))

        def allowed(u: str) -> bool:
            p = urlparse(u)
            return p.scheme in ("http", "https") and p.netloc.lower() == host and p.path.startswith(prefix) and robots.can_fetch(USER_AGENT, u)

        queue, seen, kept, texts, done, first = deque([start_url]), {start_url}, set(), set(), 0, True
        while queue:
            if done >= max_pages:
                run.stopped = "max_pages"
                break
            if run.fetched >= max_fetches:
                run.stopped = "max_fetches"
                break
            if not first:
                time.sleep(max(0.0, min(delay, deadline - time.monotonic())))
            if time.monotonic() >= deadline:
                run.stopped = "time_limit"
                break
            url = queue.popleft()
            if not first and not allowed(url):
                run.robots_skipped += 1
                continue
            run.fetched += 1
            try:
                final, html = _fetch_html(session, url, deadline=deadline, allow=None if first else allowed)
            except CrawlError:
                if first:
                    raise
                continue
            if first and _scope(final) != (host, prefix):  # the start page moved (e.g. to www.): follow it
                if urlparse(final).netloc != host:
                    robots = _load_robots(session, urlparse(final).scheme, urlparse(final).netloc, deadline)
                host, prefix = _scope(final)
                host = host.lower()
                if not robots.can_fetch(USER_AGENT, final):
                    raise CrawlError("robots")
            seen.add(final)
            title, text = _extract(html, final)
            digest = hashlib.sha1(text.encode()).hexdigest()
            if len(text.split()) >= MIN_WORDS and final not in kept and digest not in texts:
                kept.add(final)
                texts.add(digest)
                done += 1
                if on_page:
                    try:
                        on_page(done, final)
                    except Exception as err:  # the caller's own error: pass it through untouched
                        raise _CallerError(err) from err
                yield Page(final, title or final, text)
            elif len(text.split()) < MIN_WORDS and looks_js_only(html, text):
                run.js_pages += 1
                run.start_js = run.start_js or first
            for link in _links(html, final):
                p = urlparse(link)
                if p.netloc.lower() == host and p.path.startswith(prefix) and link not in seen:
                    seen.add(link)
                    queue.append(link)
            first = False
    finally:
        session.close()


class _CallerError(Exception):
    def __init__(self, err: Exception):
        super().__init__(str(err))
        self.err = err


def _guarded(pages: Iterator[Page]) -> Iterator[Page]:
    """Let CrawlError through; turn any other bug into a friendly CrawlError (details go to the log only)."""
    try:
        yield from pages
    except CrawlError:
        raise
    except _CallerError as wrapped:
        raise wrapped.err from None
    except Exception as err:  # noqa: BLE001
        log.exception("crawl failed")
        raise CrawlError("unreachable", "Something went wrong while reading that site. Try again or try another address.") from err


def crawl(start_url: str, max_pages: int = 15, on_page: Callable[[int, str], None] | None = None) -> Iterator[Page]:
    """Breadth-first crawl of pages under the start URL's folder on the same host.
    Raises only CrawlError: a bad or unsafe address is a BlockedURL; if the start page can't be read it yields nothing."""
    try:
        yield from _guarded(_crawl(start_url, max_pages, on_page, MAX_SECONDS, _Run()))
    except BlockedURL:
        raise
    except CrawlError as err:
        if err.code == "bad_url":
            raise BlockedURL(err.code, err.message) from None
        return


def read_site(url: str, max_pages: int = 15, on_page: Callable[[int, str], None] | None = None, *,
              max_seconds: float = MAX_SECONDS) -> CrawlReport:
    """Crawl and report. Raises only CrawlError (bad/unsafe address, or the start page can't be read).
    A site that answers but has no readable text returns a report with pages=[] and js_only/warnings set."""
    run = _Run()
    pages = list(_guarded(_crawl(url, max_pages, on_page, max_seconds, run)))
    js_only = run.start_js or (not pages and run.js_pages > 0)
    warnings = []
    if js_only and not pages:
        warnings.append(MESSAGES["js_only"])
    elif js_only:
        warnings.append("Some pages here are drawn with JavaScript and were skipped. Client versions use a real browser for those.")
    elif not pages:
        warnings.append(MESSAGES["no_text"])
    if run.stopped == "time_limit":
        warnings.append(f"Stopped after {max_seconds:g} seconds (demo time limit); read {len(pages)} page{'s' * (len(pages) != 1)}.")
    elif run.stopped == "max_fetches":
        warnings.append(f"Stopped after {run.fetched} tries: many pages there have little text.")
    if run.robots_skipped:
        n = run.robots_skipped
        warnings.append(f"Skipped {n} page{'s' * (n != 1)} that the site's robots.txt asks crawlers not to read.")
    return CrawlReport(run.start_url, pages, run.fetched, run.stopped, js_only, warnings)
