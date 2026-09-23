"""Polite same-site crawler: robots.txt, rate limit, size limits, and no internal addresses."""
from __future__ import annotations

import ipaddress
import socket
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterator
from urllib import robotparser
from urllib.parse import urldefrag, urljoin, urlparse

import lxml.html
import requests
import trafilatura

USER_AGENT = "SiteChatbotDemo/1.0 (portfolio demo; polite crawler)"
MAX_BYTES = 2_000_000
TIMEOUT_S = 10
DELAY_S = 0.5
SKIP_EXT = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".zip", ".gz", ".mp4", ".mp3", ".css", ".js", ".xml", ".ico", ".woff", ".woff2")


@dataclass
class Page:
    url: str
    title: str
    text: str

    @property
    def words(self) -> int:
        return len(self.text.split())


class BlockedURL(ValueError):
    pass


def check_public_url(url: str) -> None:
    """Reject anything that is not plain http(s) to a public internet address."""
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise BlockedURL("Only http:// and https:// website addresses are supported.")
    if parts.port not in (None, 80, 443):
        raise BlockedURL("Only standard web ports (80/443) are allowed.")
    try:
        infos = socket.getaddrinfo(parts.hostname, None)
    except socket.gaierror as err:
        raise BlockedURL(f"Can't find the website {parts.hostname}.") from err
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global or ip.is_multicast:
            raise BlockedURL("That address points to a private or internal network.")


def fetch(session: requests.Session, url: str, max_redirects: int = 3) -> tuple[str, str] | None:
    """Return (final_url, html) or None. Every redirect hop is re-checked."""
    for _ in range(max_redirects + 1):
        check_public_url(url)
        resp = session.get(url, timeout=TIMEOUT_S, allow_redirects=False, stream=True)
        if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
            url = urljoin(url, resp.headers.get("Location", ""))
            resp.close()
            continue
        if resp.status_code != 200 or "text/html" not in resp.headers.get("Content-Type", ""):
            resp.close()
            return None
        body = b""
        for chunk in resp.iter_content(64_000):
            body += chunk
            if len(body) > MAX_BYTES:
                resp.close()
                return None
        charset = resp.encoding if "charset" in resp.headers.get("Content-Type", "").lower() else "utf-8"
        return url, body.decode(charset or "utf-8", errors="replace")
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


def crawl(start_url: str, max_pages: int = 15, on_page: Callable[[int, str], None] | None = None) -> Iterator[Page]:
    """Breadth-first crawl of pages under the start URL's folder on the same host."""
    start_url = start_url.strip()
    if "://" not in start_url:
        start_url = "https://" + start_url
    check_public_url(start_url)
    host, prefix = _scope(start_url)

    robots = robotparser.RobotFileParser()
    try:
        robots_url = f"{urlparse(start_url).scheme}://{host}/robots.txt"
        check_public_url(robots_url)
        r = requests.get(robots_url, timeout=TIMEOUT_S, headers={"User-Agent": USER_AGENT}, allow_redirects=False)
        robots.parse(r.text.splitlines() if r.status_code == 200 else [])
    except requests.RequestException:
        robots.parse([])

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    queue, seen, done = deque([start_url]), {start_url}, 0
    while queue and done < max_pages:
        url = queue.popleft()
        if not robots.can_fetch(USER_AGENT, url):
            continue
        try:
            got = fetch(session, url)
        except (requests.RequestException, BlockedURL):
            got = None
        time.sleep(DELAY_S)
        if not got:
            continue
        final_url, html = got
        data = trafilatura.bare_extraction(html, url=final_url, with_metadata=True, as_dict=True, include_comments=False) or {}
        text = (data.get("text") or "").replace("¶", "").strip()
        if len(text.split()) >= 30:
            done += 1
            if on_page:
                on_page(done, final_url)
            yield Page(final_url, (data.get("title") or final_url).strip(), text)
        for link in _links(html, final_url):
            p = urlparse(link)
            if p.netloc == host and p.path.startswith(prefix) and link not in seen:
                seen.add(link)
                queue.append(link)
