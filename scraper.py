"""Polite same-site crawler: robots.txt, rate limit, size and time caps, no internal addresses.

Every failure a visitor can cause comes out as a CrawlError with a short, friendly message.
DNS rebinding: check_public_url() resolves the host, but the HTTP library resolves it again when it connects,
so a host could answer "public" first and "private" next. Every session used here therefore gets
PublicOnlyAdapter: right after each TCP connect (every hop, every redirect, before TLS or any byte is sent)
the address actually connected to is checked with _public_ip(), and a private one raises BlockedURL.
"""
from __future__ import annotations

import hashlib
import heapq
import ipaddress
import logging
import os
import re
import socket
import threading
import time
from collections import Counter, OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from time import monotonic as _real_monotonic  # the renderer's rate limit uses real time even when a test fakes `time`
from typing import Callable, Iterator
from urllib import robotparser
from urllib.parse import unquote, urldefrag, urljoin, urlparse, urlunparse

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
FETCH_WORKERS = 4           # pages fetched at the same time per host (1 when robots.txt asks for a Crawl-delay > DELAY_S)

# JavaScript-site fallback: a page whose HTML has almost no text is read again through Jina Reader, which runs
# the page in a browser and returns its text. Only public, in-scope, robots-allowed addresses are ever sent.
JINA_URL = "https://r.jina.ai/"
JINA_TIMEOUT_S = 20
RENDER_BELOW_WORDS = 60     # render when the HTML text has fewer words and looks like a JS shell (or is the start page)
JS_SOFT_PAGES = 6           # a JavaScript site stops at this many pages once the read has taken JS_SOFT_S
JS_SOFT_S = 15
SITEMAP_MAX_BYTES = 500_000
SITEMAP_MAX_URLS = 50
SITE_INFO_TITLE = "Contact details and site-wide info"
SITE_INFO_MAX_WORDS = 400
KEY_WORDS = ("contact", "pricing", "price", "plans", "tarif", "prix", "faq", "hours", "horaires", "locations",
             "shipping", "livraison", "about", "اتصل", "تواصل", "الأسعار")
ONE_HOP_MAX = 3             # same-host key pages outside the folder, linked from the start page
_LOW_VALUE = re.compile(r"/_modules/|genindex|modindex|/search(?:\.html)?/?$|/(?:log-?in|sign-?in|sign-?up|register|cart|"
                        r"checkout|account|settings|wp-json|feed)(?:/|$)|replytocom=", re.I)

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
    rendered: bool = False  # read through the JavaScript page renderer

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
    stopped: str            # "done" | "max_pages" | "max_fetches" | "time_limit" | "js_time" (slow JS site, enough pages)
    js_only: bool
    warnings: list[str] = field(default_factory=list)
    rendered: int = 0       # pages read through the JavaScript renderer
    jina_calls: int = 0     # renderer requests made (0 for a plain server-rendered site)


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
              deadline: float | None = None, allow: Callable[[str], bool] | None = None, timeout: float | None = None):
    """GET with every redirect hop re-checked. Returns (final_url, status, headers, body, encoding); raises CrawlError."""
    max_bytes = MAX_BYTES if max_bytes is None else max_bytes
    _harden(session)
    timeout = timeout or TIMEOUT_S
    stop_at = time.monotonic() + 3 * timeout
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
            resp = session.get(url, timeout=min(timeout, max(left, 0.5)), allow_redirects=False, stream=True)
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


def _links(html: str, base: str, with_text: bool = False) -> list:
    """Absolute <a href> links without #fragments or file downloads; (url, anchor text) pairs when with_text."""
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
                if with_text:
                    try:
                        anchor = " ".join(el.text_content().split())[:80]
                    except Exception:  # noqa: BLE001
                        anchor = ""
                    out.append((link, anchor))
                else:
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
    return clean_title(data.get("title") or ""), text


def clean_title(title: str) -> str:
    """Page title without Sphinx/MkDocs pilcrows and extra whitespace."""
    return re.sub(r"\s+", " ", (title or "").replace("¶", "")).strip()


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


# ---------------------------------------------------------------- JavaScript page renderer (Jina Reader)

class _Bucket:
    """Per-instance token bucket: at most `rpm` renderer calls per minute (Jina's free limit is 20/min)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._tokens: float | None = None
        self._at = 0.0

    def take(self, rpm: int) -> bool:
        with self._lock:
            now = _real_monotonic()
            if self._tokens is None:
                self._tokens, self._at = float(rpm), now
            self._tokens = min(float(rpm), self._tokens + (now - self._at) * rpm / 60.0)
            self._at = now
            if self._tokens >= 1:
                self._tokens -= 1
                return True
            return False


_JINA_BUCKET = _Bucket()
JINA_TOTAL = {"calls": 0}  # renderer calls made by this process (logged; 0 for server-rendered sites)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# Bounded on purpose: unbounded [^\]]* / \S+? retry from every "[" and take seconds on a hostile 40 KB line.
_MD_IMG = re.compile(r"!\[[^\]\n]{0,300}\]\([^)\n]{0,2048}\)")
_MD_LINK = re.compile(r"\[([^\]\n]{0,300})\]\(([^)\s\[]{1,2048})(?:\s+\"[^\"\n]{0,200}\")?\)")
MD_MAX_LINE = 20_000        # longer markdown lines are dropped (no real page text line is that long)
JINA_MAX_BYTES = 300_000    # renderer answers are cut here


def parse_jina(body: str) -> tuple[str, str, str]:
    """Jina Reader's text answer -> (title, url source, markdown content)."""
    head, sep, rest = (body or "").partition("Markdown Content:")
    title = source = ""
    for line in head.splitlines():
        if line.startswith("Title:"):
            title = line[6:].strip()
        elif line.startswith("URL Source:"):
            source = line[11:].strip()
    return clean_title(title), source, (rest if sep else body or "").strip()


def markdown_text(md: str) -> str:
    """Markdown -> plain text: image lines dropped, link text kept, **, __ and # marks removed."""
    lines = []
    for line in (md or "").splitlines():
        if len(line) > MD_MAX_LINE:
            continue
        line = _MD_IMG.sub("", line)
        line = _MD_LINK.sub(lambda m: m.group(1), line)
        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
        line = line.replace("**", "").replace("__", "")
        if re.fullmatch(r"\s*(=+|-{3,}|\*{3,})\s*", line):
            line = ""
        lines.append(line.rstrip())
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def markdown_links(md: str, base: str) -> list[tuple[str, str]]:
    """[text](url) links of the markdown as absolute (url, text), images and file downloads left out."""
    out = []
    for line in (md or "").splitlines():
        if len(line) > MD_MAX_LINE:
            continue
        for m in _MD_LINK.finditer(_MD_IMG.sub("", line)):
            link = urldefrag(urljoin(base, m.group(2).strip("<>")))[0]
            if urlparse(link).scheme in ("http", "https") and not link.lower().split("?")[0].endswith(SKIP_EXT):
                out.append((link, " ".join(m.group(1).split())[:80]))
    return out


def _same_site(a: str, b: str) -> bool:
    strip = lambda h: (h or "").lower().removeprefix("www.")  # noqa: E731
    return strip(urlparse(a).netloc) == strip(urlparse(b).netloc)


# ---------------------------------------------------------------- site-wide info (footer, sidebar, tel:/mailto:)

_INFO_BLOCK = re.compile(r"foot|sidebar|side-bar|widget|contact|address|hours|horaire|tags", re.I)


def _site_info_lines(html: str) -> list[str]:
    """Short lines of the footer / sidebar blocks plus tel: and mailto: links (the parts a main-text reader drops)."""
    try:
        doc = lxml.html.fromstring(html)
    except Exception:  # noqa: BLE001
        return []
    out: list[str] = []
    for el in doc.xpath("//a[starts-with(@href,'tel:') or starts-with(@href,'mailto:')]"):
        href = el.get("href", "")
        label = " ".join(el.text_content().split())
        kind = "Phone" if href.startswith("tel:") else "Email"
        value = unquote(href.split(":", 1)[1].split("?")[0]).strip()
        if value:
            out.append(f"{kind}: {value}" + (f" ({label})" if label and label != value else ""))
    blocks = doc.xpath("//footer|//aside|//*[@role='contentinfo' or @role='complementary']")
    for el in doc.xpath("//*[@class or @id]"):
        if el.tag in ("div", "section", "ul", "address", "p") and _INFO_BLOCK.search(f"{el.get('class', '')} {el.get('id', '')}"):
            blocks.append(el)
    for el in blocks:
        for bad in el.xpath(".//script|.//style|.//noscript"):
            bad.drop_tree()
        text = el.text_content()
        if len(text.split()) > 200:  # a whole content column, not a footer
            continue
        for line in text.splitlines():  # not re.split(r"\s*\n\s*"): quadratic on a long run of spaces
            line = " ".join(line.split())
            if line and len(line) <= 200:
                out.append(line)
    return out


# ---------------------------------------------------------------- crawl

def _start_folder(start_url: str) -> str:
    path = urlparse(start_url).path or "/"
    if path.endswith("/"):
        return path
    last = path.rsplit("/", 1)[-1]
    return path + "/" if "." not in last else path.rsplit("/", 1)[0] + "/"


_ABOUT_PAGES = {"about", "about-us", "aboutus", "a-propos", "qui-sommes-nous", "من-نحن"}


def _is_key(url: str, anchor: str = "") -> bool:
    """Contact, pricing, FAQ, hours, locations, shipping or about-us page, by its address or a short link text.
    "about" counts only for the about page itself ("About Git" in a docs menu is not a key page)."""
    path = unquote(urlparse(url).path).lower()
    tokens = [t for t in re.split(r"[/\-_.]+", path) if t]
    anchor = " ".join((anchor or "").lower().split())
    short = anchor if len(anchor.split()) <= 3 else ""
    for w in KEY_WORDS:
        if w == "about":
            last = re.sub(r"\.(html?|php|aspx?)$", "", path.rstrip("/").rsplit("/", 1)[-1])
            if last in _ABOUT_PAGES or short in ("about", "about us"):
                return True
        elif any(t.startswith(w) for t in tokens) or w in path or (short and w in short):
            return True
    return False


def _slug(url: str) -> tuple[str, ...]:
    last = unquote(urlparse(url).path).rstrip("/").rsplit("/", 1)[-1].lower()
    last = re.sub(r"\.(html?|php|aspx?)$", "", last)
    return tuple(t[:-1] if len(t) > 3 and t.endswith("s") else t for t in re.split(r"[-_.]+", last) if t)


# Translated copies of a site (/de/, /pt-br/, /zh-hant/...) fill a read with the same pages in other languages.
_LANG_CODES = frozenset(
    "ar az bg bn ca cs da de el en es et fa fi fil fr he hi hr hu id it ja ko lt lv ms nb nl nn pl pt ro ru sk sl sr sv "
    "th tr uk ur vi yo zh".split())
_LANG_SEG = re.compile(r"([a-z]{2,3})(?:[-_](?:[a-z]{2}|[a-z]{4}|\d{3}))?", re.I)
_HTML_LANG = re.compile(r"""<html\b[^>]{0,500}?\blang=["']?([A-Za-z]{2,3})(?:[-_][A-Za-z0-9]{2,8})?""", re.I)
_LINK_TAG = re.compile(r"<link\b[^>]{0,800}>", re.I)
_ATTR = re.compile(r"""\b(rel|hreflang|href)\s*=\s*(?:"([^"]{0,2048})"|'([^']{0,2048})'|([^\s>]{1,2048}))""", re.I)


def _path_lang(url: str) -> str:
    """Language code of the address's first path segment ("de" for /de/about/, "pt" for /pt-br/), else ""."""
    seg = urlparse(url).path.lstrip("/").split("/", 1)[0]
    m = _LANG_SEG.fullmatch(seg)
    code = m.group(1).lower() if m else ""
    return code if code in _LANG_CODES else ""


def _lang_info(html: str, base: str) -> tuple[str, list[tuple[str, str]]]:
    """(<html lang> primary subtag, [(hreflang, absolute url)] of <link rel=alternate hreflang>), from the head."""
    head = html[:200_000]
    m = _HTML_LANG.search(head)
    lang = m.group(1).lower() if m else ""
    alts = []
    for tag in _LINK_TAG.findall(head):
        attrs = {a.group(1).lower(): a.group(2) or a.group(3) or a.group(4) or "" for a in _ATTR.finditer(tag)}
        if "alternate" in attrs.get("rel", "").lower().split() and attrs.get("hreflang") and attrs.get("href"):
            alts.append((attrs["hreflang"].lower(), urldefrag(urljoin(base, attrs["href"].strip()))[0]))
        if len(alts) >= 100:
            break
    return lang, alts


class _Run:
    def __init__(self):
        self.start_url = ""
        self.fetched = 0
        self.stopped = "done"
        self.start_js = False
        self.js_pages = 0
        self.robots_skipped = 0
        self.rendered = 0
        self.jina_calls = 0
        self.jina_stop = ""      # why rendering stopped for this read ("busy" | "failed"), "" while it may go on
        self.closed = False      # the read is over (finished, failed or abandoned): workers still running stop early
        self.info: OrderedDict[str, None] = OrderedDict()
        self.lock = threading.Lock()


@dataclass
class _Got:
    url: str            # asked
    final: str = ""
    title: str = ""
    text: str = ""
    rendered: bool = False
    links: list = field(default_factory=list)  # [(url, anchor)]
    js_like: bool = False
    needed_render: bool = False
    info: list = field(default_factory=list)
    error: CrawlError | None = None
    lang: str = ""      # <html lang>, primary subtag ("en", "fr")
    alternates: list = field(default_factory=list)  # [(hreflang, url)] from <link rel="alternate" hreflang>


class _Renderer:
    """Reads one page through Jina Reader within this read's limits (JS_MAX_PAGES, JINA_RPM, stop on 429/5xx)."""

    def __init__(self, run: _Run, deadline: float):
        self.run, self.deadline = run, deadline
        self.max_pages = max(0, _env_int("JS_MAX_PAGES", 8))
        self.rpm = max(1, _env_int("JINA_RPM", 20))
        # both sessions made here, not lazily: a worker still running after close() must not open a new one
        self._session, self._fresh = self._new_session(False), self._new_session(True)

    @staticmethod
    def _new_session(fresh: bool):
        sess = requests.Session()
        sess.headers["User-Agent"] = USER_AGENT
        sess.headers["X-Retain-Images"] = "none"
        if fresh:  # skip Jina's cache: a cached copy can predate the page's scripts
            sess.headers["X-No-Cache"] = "true"
        key = os.environ.get("JINA_API_KEY", "").strip().strip("'\"")
        if key:
            sess.headers["Authorization"] = f"Bearer {key}"
        return sess

    def _sess(self, fresh: bool = False):
        return self._fresh if fresh else self._session

    def close(self):
        for attr in ("_session", "_fresh"):
            sess = getattr(self, attr, None)
            if sess is not None:
                sess.close()

    def _get(self, url: str, fresh: bool) -> tuple[int | None, bytes]:
        """(status, body) of one renderer call; status None when the renderer timed out or was unreachable."""
        if fresh:
            with self.run.lock:
                self.run.jina_calls += 1
                JINA_TOTAL["calls"] += 1
        try:
            _, status, _, body, _ = _download(self._sess(fresh), JINA_URL + url, max_redirects=0, deadline=self.deadline,
                                              timeout=JINA_TIMEOUT_S, max_bytes=JINA_MAX_BYTES, truncate=True)
            return status, body
        except CrawlError as err:
            return (None if err.code in ("timeout", "unreachable") else 0), b""

    def render(self, url: str) -> tuple[str, str, list] | None:
        """(title, text, links) of the rendered page, or None. The caller already checked scope and robots."""
        run = self.run
        with run.lock:  # jina_calls: thin pages asked twice (cached, then fresh) must not run past 2 x the page cap
            if run.closed or run.jina_stop or run.rendered >= self.max_pages or run.jina_calls >= 2 * self.max_pages:
                return None
        try:
            check_public_url(url)  # never send an internal address to the renderer
        except CrawlError:
            return None
        if not _JINA_BUCKET.take(self.rpm):
            with run.lock:
                run.jina_stop = run.jina_stop or "busy"
            return None
        with run.lock:
            run.jina_calls += 1
            JINA_TOTAL["calls"] += 1
        log.info("page renderer: %s", url)
        t0 = _real_monotonic()
        status, body = self._get(url, fresh=False)
        if status == 200 and len(markdown_text(parse_jina(_decode(body, "utf-8"))[2]).split()) < RENDER_BELOW_WORDS:
            if _JINA_BUCKET.take(self.rpm):  # thin: often a stale copy taken before the page's scripts ran: ask fresh
                status, body = self._get(url, fresh=True)
        log.info("page renderer: %s -> %s in %.1f s", url, status, _real_monotonic() - t0)
        if status is None or status == 429 or status >= 500:
            with run.lock:
                run.jina_stop = run.jina_stop or "failed"
            return None
        if status != 200:
            return None
        title, source, md = parse_jina(_decode(body, "utf-8"))
        if source and not _same_site(source, url):
            return None
        text = markdown_text(md)
        if len(text.split()) < MIN_WORDS:
            return None
        with run.lock:
            if run.rendered >= self.max_pages:
                return None
            run.rendered += 1
        return title, text, markdown_links(md, source or url)


def _fetch_one(session, renderer: _Renderer, url: str, *, first: bool, via_render: bool, deadline: float,
               allow: Callable[[str], bool] | None, in_scope: Callable[[str], bool] | None = None,
               js_site: bool = False, render_ok: Callable[[str], bool] | None = None, defer_render: bool = False) -> _Got:
    """Fetch and read one page (in a worker thread). Network problems come back in `error`; bugs raise.
    `render_ok(final)` must pass before the page goes to the renderer (default: `allow`); `defer_render` leaves
    that to the caller (the start page: its robots check and the first renderer calls come later, in parallel)."""
    got = _Got(url)
    if renderer.run.closed:  # the read ended while this fetch waited in the pool
        got.error = CrawlError("timeout")
        return got
    if via_render:  # a JavaScript site: its HTML is an empty shell, go straight to the renderer
        r = renderer.render(url)
        if r is not None:
            got.final, (got.title, got.text, got.links), got.rendered = url, r, True
            return got
    try:
        got.final, html = _fetch_html(session, url, deadline=deadline, allow=allow)
    except CrawlError as err:
        got.error = err
        return got
    got.title, got.text = _extract(html, got.final)
    got.links = _links(html, got.final, True)
    got.info = _site_info_lines(html)
    got.lang, got.alternates = _lang_info(html, got.final)
    words = len(got.text.split())
    got.js_like = words < MIN_WORDS and looks_js_only(html, got.text)
    few_links = first and in_scope is not None and sum(1 for u, _ in got.links if in_scope(u)) < 3
    got.needed_render = words < RENDER_BELOW_WORDS and (got.js_like or (first and (in_scope is None or few_links)))
    if got.needed_render and (first or js_site) and not defer_render:  # a server-rendered site's odd app page is skipped
        ok = render_ok or allow
        if ok is None or ok(got.final):
            _render_into(renderer, got)
    return got


def _render_into(renderer: _Renderer, got: _Got) -> _Got:
    """Replace the page's text with the renderer's when that has more words (links are added)."""
    words = len(got.text.split())
    r = renderer.render(got.final)
    if r is not None and len(r[1].split()) > words:
        (title, got.text, extra), got.rendered = r, True
        got.title = title or got.title
        got.links = got.links + extra
    return got


class _Done:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


# Bounded patterns, blocks cut with str.split: lazy (.*?) up to a closing tag that never comes is quadratic.
_SM_LOC = re.compile(r"<loc>\s*([^<\s][^<]{0,2047})</loc>")
_SM_PRI = re.compile(r"<priority>\s*([\d.]{1,10})\s*</priority>")
SITEMAP_MAX_BLOCKS = 200    # <url> entries read per sitemap


def _sitemap_urls(session, scheme: str, host: str, robots, deadline: float) -> list[tuple[str, float]]:
    """(url, priority) from /sitemap.xml and robots.txt Sitemap: lines (same host, one sitemap-index level)."""
    todo = [f"{scheme}://{host}/sitemap.xml"]
    for sm in (robots.site_maps() or []):
        if urlparse(sm).netloc.lower() == host and sm not in todo:
            todo.insert(0, sm)
    out: list[tuple[str, float]] = []
    seen_maps: set[str] = set()
    while todo and len(seen_maps) < 4 and time.monotonic() < deadline:
        sm = todo.pop(0)
        if sm in seen_maps:
            continue
        seen_maps.add(sm)
        try:
            final, status, _, body, _ = _download(session, sm, max_redirects=2, max_bytes=SITEMAP_MAX_BYTES, truncate=True,
                                                  deadline=deadline)
        except CrawlError:
            continue
        if status != 200 or urlparse(final).netloc.lower().removeprefix("www.") != host.removeprefix("www."):
            continue
        xml = _decode(body, "utf-8")
        if "<sitemapindex" in xml:
            n = 0
            for m in _SM_LOC.finditer(xml):
                loc = m.group(1).strip()
                if urlparse(loc).netloc.lower() == host:
                    todo.append(loc.replace("&amp;", "&"))
                n += 1
                if n >= 3:
                    break
            continue
        for block in xml.split("<url>", SITEMAP_MAX_BLOCKS + 1)[1:SITEMAP_MAX_BLOCKS + 1]:
            block = block.split("</url>", 1)[0]
            loc = _SM_LOC.search(block)
            if not loc:
                continue
            pri = _SM_PRI.search(block)
            try:
                p = float(pri.group(1)) if pri else 0.5
            except ValueError:
                p = 0.5
            out.append((loc.group(1).replace("&amp;", "&").strip(), p))
    return out


def _crawl(start_url: str, max_pages: int, on_page, max_seconds: float, run: _Run) -> Iterator[Page]:
    start_url = normalize_url(start_url)
    run.start_url = start_url
    check_public_url(start_url)
    if max_pages <= 0:
        run.stopped = "max_pages"
        return
    started = time.monotonic()
    deadline = started + max_seconds
    max_fetches = max_pages * MAX_FETCHES_FACTOR
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    renderer = _Renderer(run, deadline)
    pool = None
    try:
        scheme = urlparse(start_url).scheme
        host, prefix = _scope(start_url)
        robots = _load_robots(session, scheme, host, deadline)
        if not robots.can_fetch(USER_AGENT, start_url):
            raise CrawlError("robots")
        crawl_delay = float(robots.crawl_delay(USER_AGENT) or 0)
        delay = max(DELAY_S, min(crawl_delay, MAX_CRAWL_DELAY_S))
        workers = 1 if crawl_delay > max(DELAY_S, 0.5) else max(1, FETCH_WORKERS)
        sc = {"host": host, "prefix": prefix, "folder": _start_folder(start_url), "robots": robots}
        one_hop: set[str] = set()

        def in_scope(u: str) -> bool:
            p = urlparse(u)
            return p.scheme in ("http", "https") and p.netloc.lower() == sc["host"] and \
                (p.path.startswith(sc["prefix"]) or u in one_hop)

        def allowed(u: str) -> bool:
            return in_scope(u) and sc["robots"].can_fetch(USER_AGENT, u)

        heap: list = []
        seq = [0]
        slugs: set[tuple] = set()
        seen: set[str] = {start_url}
        js_site = [False]
        lang = {"code": "", "seg": "", "alts": set()}  # start page language, its first path segment, other-language copies

        def other_lang(link: str) -> bool:
            """A translated copy: another language's /xx/ folder, or an hreflang alternate in another language."""
            if link in lang["alts"]:
                return True
            code = _path_lang(link)
            if not code or not lang["code"] or code == lang["code"]:
                return False
            return urlparse(link).path.lstrip("/").split("/", 1)[0].lower() != lang["seg"]

        def note_alternates(g: _Got) -> None:
            for hl, alt in g.alternates:
                code = hl.split("-", 1)[0]
                if lang["code"] and code not in (lang["code"], "x") and alt != g.final and alt != run.start_url:
                    lang["alts"].add(alt)

        def push(link: str, anchor: str = "", via: bool = False, band_hint: int | None = None) -> None:
            path = urlparse(link).path
            under = (path if path.endswith("/") else path + "/").startswith(sc["folder"])
            key = _is_key(link, anchor)
            band = (0 if key else 1) if under else (2 if key else 3)
            if _LOW_VALUE.search(link):
                band = 4
            if band_hint is not None:
                band = min(band, band_hint)
            slug = _slug(link)
            refine = int(not key and any(slug[:n] in slugs for n in range(1, len(slug))))
            slugs.add(slug)
            seq[0] += 1
            heapq.heappush(heap, (band, refine, seq[0], link, via))

        def queue_links(links, final: str, first: bool) -> int:
            n = 0
            for link, anchor in links:
                if link in seen:
                    continue
                p = urlparse(link)
                if p.netloc.lower() != sc["host"] or p.scheme not in ("http", "https"):
                    continue
                if other_lang(link):
                    seen.add(link)
                    continue
                if not p.path.startswith(sc["prefix"]):
                    if not (first and len(one_hop) < ONE_HOP_MAX and _is_key(link, anchor)):
                        continue
                    one_hop.add(link)
                seen.add(link)
                push(link, anchor, via=js_site[0])
                n += sc["robots"].can_fetch(USER_AGENT, link)
            return n

        # ---- the start page (its renderer call waits until the robots check of where it landed)
        run.fetched += 1
        got = _fetch_one(session, renderer, start_url, first=True, via_render=False, deadline=deadline, allow=None,
                         in_scope=lambda u: urlparse(u).netloc.lower().removeprefix("www.") == sc["host"].removeprefix("www."),
                         defer_render=True)
        if got.error is not None:
            raise got.error
        final = got.final
        if _scope(final) != (sc["host"], sc["prefix"]):  # the start page moved (e.g. to www.): follow it
            if urlparse(final).netloc != sc["host"]:
                sc["robots"] = _load_robots(session, urlparse(final).scheme, urlparse(final).netloc, deadline)
            sc["host"], sc["prefix"] = _scope(final)
            sc["host"] = sc["host"].lower()
            sc["folder"] = _start_folder(final)
        if not sc["robots"].can_fetch(USER_AGENT, final):  # also before the renderer reads it
            raise CrawlError("robots")
        seen.add(final)
        lang["code"] = got.lang or _path_lang(final)
        lang["seg"] = urlparse(final).path.lstrip("/").split("/", 1)[0].lower()
        note_alternates(got)
        run.start_js = js_site[0] = got.js_like
        kept, texts, done = set(), set(), 0

        def accept(g: _Got) -> Page | None:
            nonlocal done
            for line in g.info:
                if len(run.info) < 400:
                    run.info.setdefault(line)
            if len(g.text.split()) < MIN_WORDS and g.info and _is_key(g.final):  # a contact form page: keep its details
                extra = list(dict.fromkeys(x for x in g.info if x not in g.text))
                g.text = (g.text + "\n" + "\n".join(extra)).strip()
            digest = hashlib.sha1(g.text.encode()).hexdigest()
            if len(g.text.split()) >= MIN_WORDS and g.final not in kept and digest not in texts:
                kept.add(g.final)
                texts.add(digest)
                done += 1
                if on_page:
                    try:
                        on_page(done, g.final)
                    except Exception as err:  # the caller's own error: pass it through untouched
                        raise _CallerError(err) from err
                return Page(g.final, g.title or g.final, g.text, g.rendered)
            if len(g.text.split()) < MIN_WORDS and g.js_like:
                run.js_pages += 1
            return None

        queued = queue_links(got.links, final, True)
        if got.needed_render or queued < 3:
            for link, pri in sorted(_sitemap_urls(session, urlparse(final).scheme, sc["host"], sc["robots"], deadline),
                                    key=lambda x: (not _is_key(x[0]), -x[1]))[:SITEMAP_MAX_URLS]:
                link = urldefrag(link)[0]
                if link not in seen and in_scope(link) and not link.lower().split("?")[0].endswith(SKIP_EXT) \
                        and not other_lang(link):
                    seen.add(link)
                    push(link, via=js_site[0], band_hint=None)

        if workers > 1:
            pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="crawl")
        inflight: deque = deque()
        if got.needed_render:  # rendered while the first other pages are read (same pool, same rate limit)
            inflight.append(pool.submit(_render_into, renderer, got) if pool is not None else _Done(_render_into(renderer, got)))
        else:
            inflight.append(_Done(got))
        start_got = got
        first_seq = True

        def js_enough() -> bool:  # a slow JavaScript site: six pages are enough once the read passes JS_SOFT_S
            return js_site[0] and done >= JS_SOFT_PAGES and time.monotonic() - started >= JS_SOFT_S

        while True:
            if done >= max_pages:
                run.stopped = "max_pages"
                break
            if js_enough():
                run.stopped = "js_time"
                break
            while heap and len(inflight) < workers and run.fetched < max_fetches and not js_enough():
                if workers == 1 and not first_seq:
                    time.sleep(max(0.0, min(delay, deadline - time.monotonic())))
                if time.monotonic() >= deadline:
                    break
                _, _, _, url, via = heapq.heappop(heap)
                if url in lang["alts"]:  # learned to be a translated copy after it was queued
                    continue
                if not allowed(url):
                    run.robots_skipped += 1
                    continue
                first_seq = False
                run.fetched += 1
                if pool is None:
                    inflight.append(_Done(_fetch_one(session, renderer, url, first=False, via_render=via,
                                                     deadline=deadline, allow=allowed, js_site=js_site[0])))
                    break  # one at a time: process it before the next fetch
                inflight.append(pool.submit(_fetch_one, session, renderer, url, first=False, via_render=via,
                                            deadline=deadline, allow=allowed, js_site=js_site[0]))
            if not inflight:
                if heap and run.fetched >= max_fetches:
                    run.stopped = "max_fetches"
                elif heap and time.monotonic() >= deadline:
                    run.stopped = "time_limit"
                break
            g = inflight.popleft().result()
            if g.error is not None:
                continue
            seen.add(g.final)
            is_start = g is start_got
            if not is_start:
                note_alternates(g)
            page = accept(g)
            queue_links(g.links, g.final, is_start)
            if page is not None:
                yield page
        if run.stopped == "done" and heap and run.fetched >= max_fetches:
            run.stopped = "max_fetches"
    finally:
        run.closed = True
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        renderer.close()
        session.close()


def _site_info_page(run: _Run, pages: list[Page], start_url: str) -> Page | None:
    """One extra page with the footer/sidebar lines and phone/email links seen on the site (deduplicated)."""
    if not pages or not run.info:
        return None
    lines, words = [], 0
    joined = "\n".join(p.text for p in pages)
    for line in run.info:
        if line in joined and not line.startswith(("Phone:", "Email:")):
            continue  # already in a page's main text
        n = len(line.split())
        if words + n > SITE_INFO_MAX_WORDS:
            break
        lines.append(line)
        words += n
    if words < 8:
        return None
    return Page(urldefrag(pages[0].url)[0] + "#site-info", SITE_INFO_TITLE, "\n".join(lines))


_TITLE_SEP = re.compile(r"\s+[|—–\-·:]\s+")


def clean_titles(pages: list[Page]) -> None:
    """Strip pilcrows and a ' | Brand' / ' — Brand' suffix that most page titles repeat (in place)."""
    for p in pages:
        p.title = clean_title(p.title)
    counts = Counter(p.title for p in pages)
    seen_titles: set[str] = set()
    for p in pages:  # a JavaScript app often gives every page the home page's title: name the others by their address
        if counts[p.title] > 1 and p.title in seen_titles:
            slug = unquote(urlparse(p.url).path).rstrip("/").rsplit("/", 1)[-1]
            slug = re.sub(r"\.(html?|php|aspx?)$", "", slug).replace("-", " ").replace("_", " ").strip()
            if slug:
                p.title = slug[:1].upper() + slug[1:]
        seen_titles.add(p.title)
    if len(pages) < 3:
        return
    tails = Counter()
    for p in pages:
        parts = _TITLE_SEP.split(p.title)
        if len(parts) > 1:
            tails[parts[-1].strip()] += 1
    if not tails:
        return
    tail, n = tails.most_common(1)[0]
    if n * 2 <= len(pages) or n < 3:
        return
    pat = re.compile(r"\s+[|—–\-·:]\s+" + re.escape(tail) + r"$")
    for p in pages:
        short = pat.sub("", p.title).strip()
        if short and short != p.title:
            p.title = short
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
              max_seconds: float = MAX_SECONDS, on_kept: Callable[[Page], None] | None = None) -> CrawlReport:
    """Crawl and report. Raises only CrawlError (bad/unsafe address, or the start page can't be read).
    A site that answers but has no readable text returns a report with pages=[] and js_only/warnings set."""
    run = _Run()
    pages = []
    try:
        for p in _guarded(_crawl(url, max_pages, on_page, max_seconds, run)):
            pages.append(p)
            if on_kept is not None:
                try:
                    on_kept(p)
                except Exception:  # noqa: BLE001 - an observer must not break the read
                    log.warning("on_kept failed", exc_info=True)
    finally:
        run.closed = True  # renderer calls still queued in worker threads give up
    clean_titles(pages)
    js_only = run.start_js or (not pages and run.js_pages > 0)
    warnings = []
    if not pages:
        warnings.append(MESSAGES["js_only"] if js_only else MESSAGES["no_text"])
    elif run.rendered:
        n = run.rendered
        warnings.append(f"{n} page{'s' * (n != 1)} here {'are' if n != 1 else 'is'} drawn with JavaScript; "
                        f"{'they were' if n != 1 else 'it was'} read through a page renderer.")
    elif js_only:
        warnings.append("Some pages here are drawn with JavaScript and were skipped. Client versions use a real browser for those.")
    if pages and run.jina_stop and (run.js_pages or run.start_js):
        warnings.append("The page renderer is busy right now, so some JavaScript pages were skipped.")
    if run.stopped == "time_limit":
        warnings.append(f"Stopped after {max_seconds:g} seconds (demo time limit); read {len(pages)} page{'s' * (len(pages) != 1)}.")
    elif run.stopped == "max_fetches":
        warnings.append(f"Stopped after {run.fetched} tries: many pages there have little text.")
    if run.robots_skipped:
        n = run.robots_skipped
        warnings.append(f"Skipped {n} page{'s' * (n != 1)} that the site's robots.txt asks crawlers not to read.")
    info = _site_info_page(run, pages, run.start_url)
    if info is not None:
        pages.append(info)
    if run.jina_calls:
        log.info("read %s: %d pages, %d rendered, %d renderer calls", run.start_url, len(pages), run.rendered, run.jina_calls)
    return CrawlReport(run.start_url, pages, run.fetched, run.stopped, js_only, warnings, run.rendered, run.jina_calls)
