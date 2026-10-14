"""Visit log for the portfolio site (slimanechanchoul.com): one event per page load, sent by the site's /hit.js to
POST /api/hit and read back with GET /api/stats (api/main.py) or scripts/visits.py.

Storage
  Upstash Redis over its REST API when KV_REST_API_URL + KV_REST_API_TOKEN (the names the Vercel Marketplace
  integration sets) or UPSTASH_REDIS_REST_URL + UPSTASH_REDIS_REST_TOKEN are set; otherwise an in-memory ring
  buffer (dev and tests: lost on restart, not shared between instances). Store calls use `requests` with short
  timeouts and swallow every failure: a broken store loses visits, it never breaks a page or the API.
  Redis keys: visits:events = JSON events, newest first (LPUSH + LTRIM to the last MAX_EVENTS);
  visits:day:<YYYY-MM-DD> = hash of daily counters "c:<country>" and "p:<path>" (expires after DAY_TTL_S).
  One hit = one pipeline request of 4 commands (LPUSH, LTRIM, 2 HINCRBY). The day key's EXPIRE is added only until
  one write of that day has succeeded on this instance: about once a day per instance.

Quota: the Upstash free tier allows 500K commands a month. HitCaps (in memory, per instance) stop at HIT_DAILY_CAP
hits a day (3,000 x 4 commands x 31 days = 372K) and HIT_IP_CAP hits per IP a day, far above real traffic. The caps
are per instance: two instances busy all day could store twice as much.

Event (what is stored): ts, utc (server time), path, qs (the page's own query string, only the QS_KEEP parameters:
hit.js filters them and make_event filters again), title, ref (referrer host only), lang, tz, sw, sh (screen),
sid (random per-tab id), ct (client clock, ms), country, region, city, ip_tz (Vercel geo headers), device, browser,
os (coarse families from the User-Agent).
Never stored: the IP address, the full User-Agent, the referrer's path or query string, other query parameters.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
import time
from collections import Counter, deque
from datetime import date, datetime, timedelta, timezone
from urllib.parse import unquote, unquote_plus, urlsplit

MAX_EVENTS = 5000
MAX_BODY = 2048            # bytes of one /api/hit body
MAX_QS = 200               # chars of the stored query string
DAY_TTL_S = 400 * 86400    # daily counters live a bit more than a year
MAX_DAYS = 400
HIT_IP_CAP = 100           # hits per IP per day, per instance (env HIT_IP_CAP)
HIT_DAILY_CAP = 3000       # hits per day, per instance (env HIT_DAILY_CAP): fits the Upstash free tier, see above
# The only query parameters kept: the ones the site's pages read, plus campaign tags. Same list as KEEP in the
# site's hit.js. Anything else (mc_eid, _hsenc, li_fat_id, email=...) is dropped.
QS_KEEP = frozenset({"site", "color", "q", "sel", "theme", "utm_source", "utm_medium", "utm_campaign"})
WRITE_TIMEOUT = (1.0, 2.0)  # (connect, read) seconds
READ_TIMEOUT = (2.0, 8.0)
EVENTS_KEY = "visits:events"
DAY_KEY = "visits:day:"
TOP = 100                  # rows per "by_*" list in /api/stats

# Crawlers, previews, monitors, headless browsers and HTTP libraries. Real in-app browsers (Instagram, Facebook,
# LinkedIn, Telegram, DuckDuckGo, Yandex app...) must pass, so their bots are caught by "bot" alone; (?<!cu) keeps
# CUBOT phones. Anything without "Mozilla/" is a tool, not a browser.
BOT_RE = re.compile(
    r"(?<!cu)bot|crawl|spider|slurp|scrap|headless|lighthouse|pagespeed|gtmetrix|preview|prerender|renderer|"
    r"google-|inspectiontool|feedfetcher|mediapartners|facebookexternalhit|facebookcatalog|meta-external|embedly|"
    r"iframely|vkshare|w3c_validator|ia_archiver|chatgpt-user|claude-(?:user|searchbot)|perplexity|cohere-ai|"
    r"anthropic-ai|curl/|wget/|python|httpx|aiohttp|go-http-client|java/|okhttp|axios|node-fetch|undici|libwww|"
    r"phantomjs|selenium|puppeteer|playwright|webdriver|cypress|pingdom|uptime|statuscake", re.I)

_CTRL = re.compile(r"[\x00-\x1f\x7f-\x9f​-‏ -‮⁦-⁩﻿]")  # incl. bidi overrides
_HOST = re.compile(r"[a-z0-9.-]{1,100}")
_LANG = re.compile(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{1,8}){0,4}")
_TZ = re.compile(r"[A-Za-z0-9_+\-/]{1,64}")
_SID = re.compile(r"[A-Za-z0-9_-]{1,40}")
_REGION = re.compile(r"[A-Za-z0-9-]{1,16}")


# ---------------------------------------------------------------- one event

def is_bot(ua: str | None) -> bool:
    ua = ua or ""
    return not ua.strip() or "mozilla/" not in ua.lower() or bool(BOT_RE.search(ua))


def ua_family(ua: str) -> tuple[str, str, str]:
    """(device, browser, os) as coarse families; the full User-Agent is never kept."""
    u = (ua or "").lower()
    if "windows" in u:
        os_ = "Windows"
    elif "iphone" in u or "ipad" in u or "ipod" in u:
        os_ = "iOS"
    elif "android" in u:
        os_ = "Android"
    elif "cros" in u:
        os_ = "ChromeOS"
    elif "macintosh" in u or "mac os x" in u:
        os_ = "macOS"
    elif "linux" in u:
        os_ = "Linux"
    else:
        os_ = "Other"
    if "ipad" in u or "tablet" in u or ("android" in u and "mobile" not in u):
        device = "tablet"
    elif "mobi" in u or "iphone" in u or "ipod" in u:
        device = "mobile"
    else:
        device = "desktop"
    for needle, name in (("edg", "Edge"), ("opr/", "Opera"), ("opera", "Opera"), ("samsungbrowser", "Samsung"),
                         ("yabrowser", "Yandex"), ("firefox/", "Firefox"), ("fxios/", "Firefox"), ("crios/", "Chrome"),
                         ("chrome/", "Chrome"), ("chromium/", "Chrome"), ("safari/", "Safari")):
        if needle in u:
            return device, name, os_
    return device, "Other", os_


def _text(v, n: int) -> str:
    return _CTRL.sub(" ", v).strip()[:n] if isinstance(v, str) else ""


def _match(v, rx: re.Pattern, n: int) -> str:
    v = v.strip()[:n] if isinstance(v, str) else ""
    return v if rx.fullmatch(v) else ""


def _int(v, lo: int, hi: int) -> int:
    try:
        x = int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0
    except (OverflowError, ValueError):  # inf / nan
        return 0
    return x if lo <= x <= hi else 0


def ref_host(ref) -> str:
    """Host of the referrer only (no path, no query string); "" for none or anything odd."""
    if not isinstance(ref, str) or not ref.strip():
        return ""
    try:
        host = (urlsplit(ref.strip()[:500]).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""
    return host if _HOST.fullmatch(host) else ""


def clean_qs(qs) -> str:
    """"?" + only the QS_KEEP parameters of a query string, each one kept as sent (same order, same encoding),
    cut to MAX_QS characters; "" when none is left."""
    if not isinstance(qs, str) or not qs.startswith("?"):
        return ""
    kept = [part for part in qs[1:].split("&") if part and unquote_plus(part.split("=", 1)[0]) in QS_KEEP]
    return ("?" + "&".join(kept))[:MAX_QS] if kept else ""


def make_event(body, headers, ua: str, now: float) -> dict | None:
    """The stored event for one /api/hit body, or None when the body is not a hit. `headers` = request headers
    (Vercel geo headers: x-vercel-ip-country, -country-region, -city (URL-encoded), -timezone)."""
    if not isinstance(body, dict):
        return None
    path = _text(body.get("path"), 200)
    if not path.startswith("/"):
        return None
    qs = clean_qs(_text(body.get("qs"), MAX_BODY))  # filtered before the cut: junk cannot push a kept one out
    cc = (headers.get("x-vercel-ip-country") or "").strip().upper()
    device, browser, os_ = ua_family(ua)
    return {
        "ts": int(now),
        "utc": datetime.fromtimestamp(int(now), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "path": path,
        "qs": qs,
        "title": _text(body.get("title"), 120),
        "ref": ref_host(body.get("ref")),
        "lang": _match(body.get("lang"), _LANG, 35),
        "tz": _match(body.get("tz"), _TZ, 64),
        "sw": _int(body.get("sw"), 0, 20000),
        "sh": _int(body.get("sh"), 0, 20000),
        "sid": _match(body.get("sid"), _SID, 40),
        "ct": _int(body.get("t"), 10**12, 10**13),
        "country": cc if re.fullmatch(r"[A-Z]{2}", cc) else "",
        "region": _match(headers.get("x-vercel-ip-country-region"), _REGION, 16),
        "city": _text(unquote(headers.get("x-vercel-ip-city") or ""), 80),
        "ip_tz": _match(headers.get("x-vercel-ip-timezone"), _TZ, 64),
        "device": device, "browser": browser, "os": os_,
    }


def counter_fields(event: dict) -> list[str]:
    return ["c:" + (event.get("country") or "??"), "p:" + event["path"]]


# ---------------------------------------------------------------- stores

class MemoryStore:
    """Ring buffer of the last MAX_EVENTS events and per-day counters, in this process only."""
    kind = "memory"

    def __init__(self, max_events: int = MAX_EVENTS):
        self.max_events = max_events
        self._lock = threading.Lock()
        self._events: deque[str] = deque(maxlen=max_events)  # newest at the left, like LPUSH
        self._days: dict[str, Counter] = {}

    def add(self, event: dict) -> bool:
        day = event["utc"][:10]
        with self._lock:
            self._events.appendleft(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
            c = self._days.setdefault(day, Counter())
            for f in counter_fields(event):
                c[f] += 1
            while len(self._days) > MAX_DAYS + 5:
                del self._days[min(self._days)]
        return True

    def lrange(self, start: int, stop: int) -> list[str] | None:
        with self._lock:
            return list(self._events)[start:stop + 1]

    def day_hashes(self, days: list[str]) -> list[dict[str, int]] | None:
        with self._lock:
            return [dict(self._days.get(d, {})) for d in days]


class UpstashStore:
    """Upstash Redis through its REST pipeline endpoint. Every call returns None (or False) on any failure."""
    kind = "upstash"

    def __init__(self, url: str, token: str, post=None, max_events: int = MAX_EVENTS):
        self.url = url.rstrip("/")
        self._token = token
        self._post = post
        self.max_events = max_events
        self._lock = threading.Lock()
        self._ttl_days: set[str] = set()  # day keys whose EXPIRE this instance has sent (the last 3 days)

    def __repr__(self) -> str:  # never the token
        return f"UpstashStore({self.url!r})"

    def _pipeline(self, commands: list[list[str]], timeout) -> list | None:
        try:
            post = self._post
            if post is None:
                import requests
                post = requests.post
            r = post(self.url + "/pipeline", json=commands, timeout=timeout,
                     headers={"Authorization": "Bearer " + self._token})
            if r.status_code != 200:
                return None
            out = r.json()
            if not isinstance(out, list) or len(out) != len(commands):
                return None
            if any(not isinstance(x, dict) or x.get("error") is not None for x in out):
                return None
            return [x.get("result") for x in out]
        except Exception:  # noqa: BLE001 - a broken store must never break a page
            return None

    def add(self, event: dict) -> bool:
        day_key = DAY_KEY + event["utc"][:10]
        cmds = [["LPUSH", EVENTS_KEY, json.dumps(event, ensure_ascii=False, separators=(",", ":"))],
                ["LTRIM", EVENTS_KEY, "0", str(self.max_events - 1)]]
        cmds += [["HINCRBY", day_key, f, "1"] for f in counter_fields(event)]
        with self._lock:
            expire = day_key not in self._ttl_days
        if expire:  # the TTL only has to be set once; a failed write sends it again next time
            cmds.append(["EXPIRE", day_key, str(DAY_TTL_S)])
        ok = self._pipeline(cmds, WRITE_TIMEOUT) is not None
        if ok and expire:
            with self._lock:
                self._ttl_days = set(sorted(self._ttl_days | {day_key})[-3:])  # keys sort by date
        return ok

    def lrange(self, start: int, stop: int) -> list[str] | None:
        out = self._pipeline([["LRANGE", EVENTS_KEY, str(start), str(stop)]], READ_TIMEOUT)
        if out is None or not isinstance(out[0], list):
            return None
        return [x for x in out[0] if isinstance(x, str)]

    def day_hashes(self, days: list[str]) -> list[dict[str, int]] | None:
        out = self._pipeline([["HGETALL", DAY_KEY + d] for d in days], READ_TIMEOUT) if days else []
        if out is None:
            return None
        hashes = []
        for flat in out:  # HGETALL over REST: [field, value, field, value, ...]
            h: dict[str, int] = {}
            if isinstance(flat, list):
                for f, v in zip(flat[::2], flat[1::2]):
                    try:
                        h[str(f)] = int(v)
                    except (TypeError, ValueError):
                        pass
            hashes.append(h)
        return hashes


def _clean_env(v: str | None) -> str:
    return (v or "").strip().strip("'\"").strip()


def from_env(env=None) -> MemoryStore | UpstashStore:
    env = os.environ if env is None else env
    for url_name, token_name in (("KV_REST_API_URL", "KV_REST_API_TOKEN"),
                                 ("UPSTASH_REDIS_REST_URL", "UPSTASH_REDIS_REST_TOKEN")):
        url, token = _clean_env(env.get(url_name)), _clean_env(env.get(token_name))
        if url.startswith(("https://", "http://")) and token:
            return UpstashStore(url, token)
    return MemoryStore()


def safe_add(store, event: dict) -> None:
    """Background task after the 204: whatever happens, nothing reaches the visitor."""
    try:
        store.add(event)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------- caps

class HitCaps:
    """Per-IP and global daily caps on hits, in memory per instance; reset at 00:00 UTC. The IP is kept only as a
    salted hash (random salt per process), and only for the day's count."""

    def __init__(self, ip_cap: int = HIT_IP_CAP, daily_cap: int = HIT_DAILY_CAP, clock=time.time):
        self.ip_cap, self.daily_cap, self._clock = ip_cap, daily_cap, clock
        self._lock = threading.Lock()
        self._salt = secrets.token_bytes(16)
        self._day, self._total, self._by_ip = "", 0, {}

    def take(self, ip: str) -> bool:
        h = hashlib.blake2b((ip or "").encode(), key=self._salt, digest_size=8).digest()
        with self._lock:
            day = time.strftime("%Y-%m-%d", time.gmtime(self._clock()))
            if day != self._day:
                self._day, self._total, self._by_ip = day, 0, {}
            if self._total >= self.daily_cap or self._by_ip.get(h, 0) >= self.ip_cap:
                return False
            self._by_ip[h] = self._by_ip.get(h, 0) + 1
            self._total += 1
            return True


# ---------------------------------------------------------------- stats

def _top(c: Counter) -> list[dict]:
    return [{"key": k, "n": n} for k, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP]]


def stats(store, now: float, days: int = 7, limit: int = 200, chunk: int = 1000) -> dict | None:
    """Visits of the last `days` UTC days (today included). by_day/by_country/by_page come from the daily
    counters (exact even after old events are trimmed); events, sessions, by_city and by_ref come from the stored
    events of the period (`partial` = the log is full and may not reach back to the period start). None when the
    store cannot be read."""
    days = max(1, min(MAX_DAYS, int(days)))
    limit = max(0, min(MAX_EVENTS, int(limit)))
    today = datetime.fromtimestamp(now, timezone.utc).date()
    day_list = [(today - timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]
    since = datetime.combine(date.fromisoformat(day_list[0]), datetime.min.time(), timezone.utc).timestamp()
    hashes = store.day_hashes(day_list)
    if hashes is None:
        return None
    by_day, by_country, by_page = [], Counter(), Counter()
    for d, h in zip(day_list, hashes):
        n = 0
        for f, v in h.items():
            if f.startswith("c:"):
                by_country[f[2:] or "??"] += v
                n += v
            elif f.startswith("p:"):
                by_page[f[2:]] += v
        by_day.append({"day": d, "views": n, "sessions": 0})

    events, start, older, seen = [], 0, False, 0
    cap = getattr(store, "max_events", MAX_EVENTS)
    while start < cap and not older:  # newest first, in chunks, until an event is older than the period
        want = min(chunk, cap - start)
        raw = store.lrange(start, start + want - 1)
        if raw is None:
            return None
        seen += len(raw)
        for item in raw:
            try:
                ev = json.loads(item)
                ts = int(ev.get("ts") or 0) if isinstance(ev, dict) else None
            except (TypeError, ValueError):
                ts = None
            if ts is None:
                continue
            if ts < since:
                older = True
                break
            events.append(ev)
        if len(raw) < want:
            break
        start += want
    partial = not older and seen >= cap  # the log is full and still inside the period: older events were trimmed

    sessions_by_day: dict[str, set] = {}
    by_city, by_ref = Counter(), Counter()
    for ev in events:
        cc = ev.get("country") or "??"
        by_city[f"{ev.get('city') or '(unknown)'}, {cc}"] += 1
        by_ref[ev.get("ref") or "(direct)"] += 1
        sessions_by_day.setdefault(str(ev.get("utc", ""))[:10], set()).add(ev.get("sid") or f"ts:{ev.get('ts')}")
    for row in by_day:
        row["sessions"] = len(sessions_by_day.get(row["day"], ()))
    return {
        "store": getattr(store, "kind", "?"), "days": days, "from": day_list[0], "to": day_list[-1],
        "generated": datetime.fromtimestamp(int(now), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "views": sum(r["views"] for r in by_day),
        "sessions": len({ev.get("sid") for ev in events if ev.get("sid")}),
        "by_day": by_day, "by_country": _top(by_country), "by_page": _top(by_page), "by_city": _top(by_city),
        "by_ref": _top(by_ref), "events": events[:limit], "events_in_period": len(events), "partial": partial,
    }
