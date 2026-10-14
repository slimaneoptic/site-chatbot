"""HTTP API for the two front ends (Demo 1 "Ask any website", Demo 2 widget/studio).

Run from the repo root:  uvicorn api.main:app --port 8000

Endpoints
  GET  /healthz                    -> {ok, ai, kbs, budget, inst (8-hex id of this instance), up_s, ai_key
                                      ("ok" | "missing" | "rejected" | "unchecked"), restore (KB_SECRET set)}
  GET  /api/examples               -> pre-indexed example sources (eval/fixtures) + starter questions,
                                      with the cached answer of each starter when the shared cache has one
  POST /api/kb                     -> JSON {url, max_pages?} or multipart PDFs (field "files")
                                      -> {kb_id, label, kind, url, lang, pages, words, warnings,
                                          page_list:[{title, url, rendered}] (first 20 pages, crawl order),
                                          starters:[0-4 checked questions in the site's language], rendered:int,
                                          snapshot (only when the body has snapshot:true and the pages are <= 900 KB:
                                          opaque and signed; the front sends it back unchanged as `restore`)}
       Any JSON body may be sent as text/plain (no CORS preflight).
       Daily per-IP caps: site reads IP_CRAWL_CAP (429 error code ip_crawl_cap), PDF uploads IP_UPLOAD_CAP
       (429 error code ip_upload_cap).
       Crawl progress: send the same POST with ?stream=1 (or Accept: text/event-stream). The answer is then
       an SSE stream: progress{n, url} (one per page read) ... then done{<the JSON above>} or error{code, message}.
  POST /api/chat {kb_id, messages:[{role, content}], session_id, restore?}
       restore = the snapshot from /api/kb, sent only when retrying after kb_expired: a valid one rebuilds the KB
       under the same kb_id (no crawl, no AI; IP_RESTORE_CAP per IP per day -> ip_restore_cap), an invalid one
       gives kb_expired again. Bodies over 256 KB are accepted only with restore, up to 1 MB (else too_big).
       -> SSE, always in this order: meta{lang, dir, pages} . sources[{n, title, url, passage, hl, hl_at, terms}]
          (hl = best sentence of the passage, hl_at = [start, end] of hl in passage as JavaScript string
          indices (UTF-16 code units) or null)
          . [notice{code, message}] . delta{text} (many) . followups[...] . done{ttfw_ms, cached, cost,
          left_today, status, cited, model, hl}
          hl in sources is a guess from the question (or from the cached answer); done.hl = {"<n>": {hl, hl_at}}
          is the sentence of each cited source that the answer's [n] sentences overlap most: the page moves the
          highlight there. followups only holds questions the local search can answer on this site.
          Fatal problems send only error{code, message} (codes: kb_expired, bad_request, server_error).
          kb_expired: the client re-reads the site from its saved URL (POST /api/kb) and asks again.
          notice codes (non-fatal, the answer still follows as passages): session_cap, ip_cap,
          global_budget, ai_busy, ai_unavailable (the key is missing or rejected).
          done.closest = {title, url} of the nearest page when the site does not answer (else null).
          Pinned examples: an exact cached question (starter or follow-up chip) is served from the cache mid-chat.
  POST /api/extract {kb_id, fields}  -> {kb_id, fields, rows:[{<label>: value, ..., source:{url, title}}],
                                         pages_read, warnings} (or {error:{code, message}})
       Multipart uploads are counted while they stream: over MAX_UPLOAD_MB -> 413 too_big even without a
       Content-Length header.
  GET  /api/kb/{id}.json           -> pages (text only) + the last extracted rows
  GET  /api/kb/{id}.csv            -> the last extracted rows as CSV (or the page list when nothing was extracted)
  POST /api/hit                    -> 204. One page view of slimanechanchoul.com, sent by the site's /hit.js
       (sendBeacon, text/plain or JSON, max 2 KB): {path, qs, title, ref, lang, tz, sw, sh, sid, t}. Needs an
       allowed Origin header (no Origin -> 403). The server adds UTC time, country/region/city from Vercel's geo
       headers, coarse device/browser/os from the User-Agent, keeps only the referrer's host and only the qs
       parameters in visits.QS_KEEP. Never stored: the IP and the full User-Agent. Bots are dropped (204, nothing
       stored). Caps, both per instance (429): HIT_IP_CAP (100) hits per IP per day and HIT_DAILY_CAP (3000) hits
       per day, which keeps a month of fake hits inside the Upstash free tier (visits.py docstring). The Origin
       check stops other sites' pages, not scripts. Storage: visits.py (Upstash Redis or memory).
       Leave summary, at most one more body per page view (max 4 KB): {type: "leave", page, sid, pv, active_ms,
       max_scroll_pct, clicks: [{l, ms, n?}] (25 max)} -> 204, stored apart (no IP, no geo, no device), same Origin,
       bot and cap rules; 400 when page, sid or the clicks list is not valid. A body without type "leave" is a page
       view, handled exactly as before (2 KB max).
  POST /api/answer                 -> answers from passages a WordPress site sent (the Site Answers plugin's online
       demo on WordPress Playground). JSON (any Content-Type, max 20 KB): {question (1-500 chars),
       passages: [{n, url (http/https), title, text (max 1,500 chars)}] (1-6), history? ([{role, content}], max 3
       turns, 1,000 chars each), site_label?, site_lang?, session_id?, translate_query? (bool)}
       -> {answer, sources: [{n, title, url, passage}] (the cited passages; all of them when none is cited),
           mode ("ai" | "extractive"), not_found (the answer says the site does not cover it), cited, notice}.
       Its own caps, apart from /api/chat's so the WordPress demo can never use up /ask and /assistant:
       WP_SESSION_CAP (12) per session, WP_IP_CAP (15) per IP (an IPv6 address counts as its /64) and WP_DAILY_USD
       (0.25) per instance and day; answers are cut at WP_MAX_TOKENS (500) output tokens. Over a cap -> mode
       "extractive" with the caps' friendly message as the answer and every passage as a source.
       A "could not find" sentence cites nothing: its [n] marks are removed before the sources are chosen.
       translate_query: true (with site_lang, no passages needed) -> {query}: the question as a short search query in
       site_lang (unchanged when the AI is off or a cap is reached); it counts toward the same caps as an answer.
       Optional hint (max 900 chars): words the site uses, preferred by the translation; cut to a plain word list
       (at most 60 words) before it reaches the model.
       Origins: this route also accepts WP_DEMO_ORIGINS; every other POST refuses them.
  GET  /api/stats?days=7&limit=200 -> visit totals by day, country, city, page and referrer host + the newest events.
       Each event also has active_ms (time on page while visible and focused), scroll (% of the page reached) and
       clicks [{l, ms, n?}] from its leave summary (None, None, [] without one); engagement = per page: n
       summaries, median_ms, clicks, top labels; leaves_in_period (None when the summaries cannot be read).
       Header X-Stats-Key must equal STATS_KEY (401 otherwise); without STATS_KEY the route answers 404.
       503 store_unavailable when the store cannot be read. CLI: scripts/visits.py.

Env: CORS_ORIGINS (comma list, default https://slimanechanchoul.com), CORS_ORIGIN_REGEX (default any
http://localhost:* / 127.0.0.1:*); a POST whose Origin header matches neither gets 403 origin_not_allowed (text/plain
bodies skip the CORS preflight, so CORS alone would not stop another site's page from spending the caps), SESSION_CAP, IP_CAP, DAILY_USD, IP_CRAWL_CAP, IP_UPLOAD_CAP (10 PDF uploads per IP per day), KB_TTL_S (3600),
KB_MAX (40), MAX_PAGES (15), EXTRACT_MAX_PAGES (5), MAX_UPLOAD_MB (20), STARTER_ANSWERS (json file),
TRUSTED_PROXY_HOPS (1: the per-IP caps use the X-Forwarded-For entry our proxy added, never the visitor's own),
CLIENT_IP_HEADER (optional platform header with the visitor address), OPENROUTER_API_KEY (spaces/quotes trimmed),
KB_SECRET (32+ random bytes; needed for restore across instances), IP_RESTORE_CAP (60), JINA_API_KEY (optional),
JINA_RPM (20), JS_MAX_PAGES (8; rendered pages count inside MAX_PAGES).
WP_DEMO_ORIGINS (comma list, default https://playground.wordpress.net): origins allowed on POST /api/answer only
(CORS preflight included), for the WordPress plugin's public demo. WP_SESSION_CAP, WP_IP_CAP, WP_DAILY_USD,
WP_MAX_TOKENS: the caps of POST /api/answer (see above). The caps live in each instance's memory: the AI key's own
credit limit at OpenRouter is the only hard stop across instances.
Visit log: KV_REST_API_URL + KV_REST_API_TOKEN (or UPSTASH_REDIS_REST_URL + UPSTASH_REDIS_REST_TOKEN; none = in
memory), STATS_KEY, HIT_IP_CAP (100), HIT_DAILY_CAP (3000) (both per instance).
"""
from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import os
import secrets
import queue
import re
import sys
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, Response, StreamingResponse  # noqa: E402
from starlette.background import BackgroundTask  # noqa: E402

import answer  # noqa: E402
import budget as budget_mod  # noqa: E402
import cache as cache_mod  # noqa: E402
import docs  # noqa: E402
import extract  # noqa: E402
import llm  # noqa: E402
import scraper  # noqa: E402
import search  # noqa: E402
import starters as starters_mod  # noqa: E402
import visits as visits_mod  # noqa: E402
from scraper import Page  # noqa: E402

FIXTURES = ROOT / "eval" / "fixtures"
QUESTIONS = ROOT / "eval" / "questions.jsonl"
EXAMPLES = {  # fixture name -> label shown on the page
    "shop": "Atlas Home (shop help pages)",
    "shop_ar": "سوق الدار (متجر، بالعربية)",
    "python_tutorial": "The Python Tutorial",
    "fastapi_tutorial": "FastAPI tutorial",
}
STARTERS_PER_EXAMPLE = 4
EXTRA_STARTERS = {  # added after the eval starters (eval/questions.jsonl has only one first-turn Arabic question)
    "shop_ar": ["كم يستغرق التوصيل؟", "هل يمكنني الدفع عند الاستلام؟", "ما هو رقم الواتساب للتواصل معكم؟"],
}
PAGE_LIST_MAX = 20  # page_list in the /api/kb answer: first pages in crawl order
MAX_QUESTION_CHARS = 1000
MAX_HISTORY = 10
MAX_HISTORY_CHARS = 4000  # per message; answer.py trims further for the model

# POST /api/answer (WordPress plugin demo): small bodies, a handful of passages chosen by the site itself.
WP_PATHS = ("/api/answer",)
WP_MAX_BYTES = 20_000
WP_QUESTION_CHARS = 500
WP_MAX_PASSAGES = 6
WP_PASSAGE_CHARS = 1500
WP_HISTORY_MESSAGES = 6  # three turns: a question and its answer each
WP_HISTORY_CHARS = 1000
WP_LABEL_CHARS = 100
WP_HINT_WORDS = 60  # the translation hint is cut to a plain word list: words, not instructions

MESSAGES = {
    "kb_expired": "This site was read more than an hour ago and has been cleared. Reading it again now.",
    "bad_request": "Something was missing in that request. Please try again.",
    "origin_not_allowed": "This API only answers requests from the demo pages it was built for.",
    "server_error": "Something went wrong on our side. Please try again in a minute.",
    "ai_busy": "The AI is busy right now, so here are the most relevant passages from the site instead.",
    "ai_unavailable": "The AI is switched off on this demo right now, so here are the most relevant passages from the site instead.",
    "ip_restore_cap": "This network has reopened many sites today. Read the site again, or come back tomorrow.",
    "extract_busy": "The AI is busy right now, so extraction stopped early. Please try again in a minute.",
    "no_file": "Add a PDF file or a website address.",
    "too_big": "That upload is too big for the demo.",
    "no_text": scraper.MESSAGES["no_text"],
    "no_fields": "Name the fields to extract, separated by commas (for example: product, price).",
    "no_ai": "Extraction needs the AI, which is not available right now.",
    "hit_cap": "Too many page views from this network today.",
    "unauthorized": "Wrong or missing X-Stats-Key header.",
    "store_unavailable": "The visit log cannot be read right now.",
}
_REASON_BY_MESSAGE = {v: k for k, v in budget_mod.MESSAGES.items()}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


# ---------------------------------------------------------------- knowledge bases

@dataclass
class KB:
    id: str
    label: str
    kind: str  # "site" | "pdf" | "example"
    pages: list[Page]
    index: search.Index
    source: str  # cache.source_id(pages)
    start_url: str = ""
    warnings: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)  # last extraction
    fields: list[str] = field(default_factory=list)
    starters: list[str] = field(default_factory=list)  # checked starter questions in the site's language
    rendered: int = 0  # pages read through the JavaScript page renderer

    @property
    def words(self) -> int:
        return sum(p.words for p in self.pages)

    def summary(self, snapshot: dict | None = None) -> dict:
        out = {"kb_id": self.id, "label": self.label, "kind": self.kind, "url": self.start_url,
               "lang": self.index.lang, "pages": len(self.pages), "words": self.words, "warnings": self.warnings,
               "page_list": [{"title": p.title, "url": p.url, "rendered": bool(getattr(p, "rendered", False))}
                             for p in self.pages[:PAGE_LIST_MAX]],
               "starters": list(self.starters), "rendered": self.rendered}
        if snapshot is not None:
            out["snapshot"] = snapshot
        return out


def make_kb(pages: list[Page], label: str, kind: str, *, kb_id: str | None = None, start_url: str = "",
            warnings: list[str] | None = None) -> KB:
    return KB(kb_id or uuid.uuid4().hex[:12], label, kind, pages, search.Index(pages), cache_mod.source_id(pages),
              start_url, list(warnings or []))


# ---------------------------------------------------------------- signed snapshots (restore on another instance)

SNAPSHOT_MAX_BYTES = 900_000   # pages serialized above this: no snapshot
SNAPSHOT_TTL_S = 24 * 3600
SNAPSHOT_MAX_PAGES = 23
MAX_RESTORE_BYTES = 1_000_000  # chat / extract body that carries a snapshot
_PROCESS_SECRET = secrets.token_bytes(32)  # without KB_SECRET a snapshot only works on the instance that made it


def _kb_secret() -> bytes:
    env = os.environ.get("KB_SECRET", "")
    return env.encode() if env else _PROCESS_SECRET


def _snap_canon(label: str, kind: str, url: str, pages: list) -> bytes:
    return json.dumps([label, kind, url, pages], ensure_ascii=False, separators=(",", ":")).encode()


def _snap_sig(v: int, kb_id: str, exp: int, canon: bytes) -> str:
    msg = f"{v}|{kb_id}|{exp}|{hashlib.sha256(canon).hexdigest()}".encode()
    return hmac.new(_kb_secret(), msg, hashlib.sha256).hexdigest()


def make_snapshot(kb: "KB", now: float) -> dict | None:
    """Opaque signed copy of a KB's pages the front keeps and sends back after kb_expired; None when too big."""
    pages = [{"url": p.url, "title": p.title, "text": p.text, "rendered": bool(getattr(p, "rendered", False))}
             for p in kb.pages[:SNAPSHOT_MAX_PAGES]]
    canon = _snap_canon(kb.label, kind := kb.kind, kb.start_url, pages)
    if len(canon) > SNAPSHOT_MAX_BYTES:
        return None
    exp = int(now) + SNAPSHOT_TTL_S
    return {"v": 1, "kb_id": kb.id, "label": kb.label, "kind": kind, "url": kb.start_url, "exp": exp, "pages": pages,
            "sig": _snap_sig(1, kb.id, exp, canon)}


def verify_snapshot(snap, kb_id: str, now: float) -> list[Page] | None:
    """The snapshot's pages when its signature, kb_id and expiry are valid; None otherwise."""
    try:
        if not isinstance(snap, dict) or snap.get("v") != 1 or str(snap.get("kb_id")) != kb_id or not kb_id:
            return None
        exp = int(snap["exp"])
        if exp < now or exp > now + SNAPSHOT_TTL_S + 60:
            return None
        pages = snap["pages"]
        if not isinstance(pages, list) or not pages or len(pages) > SNAPSHOT_MAX_PAGES:
            return None
        canon = _snap_canon(snap["label"], snap["kind"], snap["url"], pages)
        if not hmac.compare_digest(_snap_sig(1, kb_id, exp, canon), str(snap.get("sig", ""))):
            return None
        return [Page(str(p["url"]), str(p["title"]), str(p["text"]), bool(p.get("rendered"))) for p in pages]
    except (KeyError, TypeError, ValueError):
        return None


class RestoreCap:
    """Per-IP daily count of KB restores (IP_RESTORE_CAP)."""

    def __init__(self, cap: int, clock=time.time):
        self.cap, self._clock, self._lock = cap, clock, threading.Lock()
        self._day, self._by_ip = "", {}

    def take(self, ip: str) -> bool:
        with self._lock:
            day = time.strftime("%Y-%m-%d", time.gmtime(self._clock()))
            if day != self._day:
                self._day, self._by_ip = day, {}
            if self._by_ip.get(ip, 0) >= self.cap:
                return False
            self._by_ip[ip] = self._by_ip.get(ip, 0) + 1
            return True


class KBStore:
    """In-memory LRU of indexes; an entry expires `ttl_s` after its last use. Pinned (example) entries never expire."""

    def __init__(self, max_entries: int = 40, ttl_s: int = 3600, clock=time.time):
        self.max_entries, self.ttl_s, self._clock = max_entries, ttl_s, clock
        self._lock = threading.Lock()
        self._items: OrderedDict[str, tuple[float, KB]] = OrderedDict()
        self._pinned: dict[str, KB] = {}

    def pin(self, kb: KB) -> None:
        self._pinned[kb.id] = kb

    def put(self, kb: KB) -> None:
        with self._lock:
            self._items[kb.id] = (self._clock(), kb)
            self._items.move_to_end(kb.id)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def get(self, kb_id: str) -> KB | None:
        if kb_id in self._pinned:
            return self._pinned[kb_id]
        with self._lock:
            item = self._items.get(kb_id)
            if item is None:
                return None
            if self._clock() - item[0] > self.ttl_s:
                del self._items[kb_id]
                return None
            self._items[kb_id] = (self._clock(), item[1])
            self._items.move_to_end(kb_id)
            return item[1]

    def __len__(self) -> int:
        return len(self._items) + len(self._pinned)


def _url_label(url: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", url).rstrip("/") or url


def load_examples(store: KBStore) -> list[dict]:
    starters: dict[str, list[str]] = {}
    if QUESTIONS.exists():
        for line in QUESTIONS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("answerable") and r.get("prev") is None:
                starters.setdefault(r["fixture"], []).append(r["question"])
    out = []
    for name, label in EXAMPLES.items():
        path = FIXTURES / f"{name}.json"
        if not path.exists():
            continue
        pages = [Page(p["url"], p["title"], p["text"]) for p in json.loads(path.read_text(encoding="utf-8"))]
        kb = make_kb(pages, label, "example", kb_id=f"ex-{name}", start_url=pages[0].url if pages else "")
        store.pin(kb)
        qs = list(dict.fromkeys(starters.get(name, []) + EXTRA_STARTERS.get(name, [])))
        out.append({"kb": kb, "starters": qs[:STARTERS_PER_EXAMPLE]})
    return out


# ---------------------------------------------------------------- SSE helpers

def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def sse_response(gen: Iterator[str]) -> StreamingResponse:
    return StreamingResponse(gen, media_type="text/event-stream", headers=SSE_HEADERS)


def client_ip(request: Request) -> str:
    """Visitor address for the daily caps. The leftmost X-Forwarded-For entry is whatever the visitor sent, so
    faking it would dodge the per-IP caps; instead take the entry written by our own proxy: the
    TRUSTED_PROXY_HOPS-th from the right (default 1 = the address Render's proxy saw). CLIENT_IP_HEADER names a
    header the platform sets itself and overwrites (e.g. cf-connecting-ip); when set and present it wins."""
    own = os.environ.get("CLIENT_IP_HEADER", "").strip().lower()
    return budget_mod.pick_client_ip(request.headers.getlist("x-forwarded-for"),
                                     request.client.host if request.client else None,
                                     max(0, _env_int("TRUSTED_PROXY_HOPS", 1)),
                                     request.headers.get(own, "") if own else None)


MAX_JSON_BYTES = 256_000  # chat / extract / kb-by-url bodies are tiny; refuse anything bigger before reading it


class _TooBig(Exception):
    pass


async def _limited_form(request: Request, max_bytes: int):
    """request.form() that stops reading (raises _TooBig) once the body passes max_bytes, whatever the headers say."""
    from starlette.formparsers import MultiPartParser

    async def body():
        n = 0
        async for chunk in request.stream():
            n += len(chunk)
            if n > max_bytes:
                raise _TooBig
            yield chunk

    return await MultiPartParser(request.headers, body()).parse()


async def _read_json(request: Request) -> dict:
    """The JSON body (any Content-Type, e.g. text/plain to skip the CORS preflight). Bodies above MAX_JSON_BYTES are
    accepted only when they carry a `restore` snapshot, up to MAX_RESTORE_BYTES. Raises _TooBig or ValueError."""
    if _declared_size(request) > MAX_RESTORE_BYTES:
        raise _TooBig
    raw = b""
    async for chunk in request.stream():
        raw += chunk
        if len(raw) > MAX_RESTORE_BYTES:
            raise _TooBig
    body = json.loads(raw.decode("utf-8") or "null")
    if not isinstance(body, dict):
        raise ValueError("not an object")
    if len(raw) > MAX_JSON_BYTES and not body.get("restore"):
        raise _TooBig
    return body


def _declared_size(request: Request) -> int:
    try:
        return int(request.headers.get("content-length") or 0)
    except ValueError:
        return 0


def _err(code: str, message: str | None = None, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message or MESSAGES.get(code, MESSAGES["server_error"])}},
                        status_code=status)


# ---------------------------------------------------------------- sources: highlight

_SENT_SPLIT = re.compile(r"(?<=[.!?؟])\s+|\n+")


def _highlight(p: search.Passage, query: str) -> tuple[str, list[str]]:
    """Best sentence of the passage for the query (shown blue-soft) and the matching surface words.
    Only a guess before the answer exists: `_answer_hl` re-picks it from the answer that cites the passage."""
    q = set(search._content(search._tokens(query)))
    if not q:
        return "", []
    terms: list[str] = []
    for w in re.findall(r"\w+", p.text):
        if w not in terms and set(search._tokens(w)) & q:
            terms.append(w)
    best, best_n = "", 0
    for s in _SENT_SPLIT.split(p.text):
        n = len(set(search._tokens(s)) & q)
        if n > best_n:
            best, best_n = s.strip(), n
    return best, terms[:12]


def _cited_text(answer_text: str, n: int) -> str:
    """The answer sentences that cite [n] (marks removed); the whole answer when none cites it."""
    out = []
    for s in re.split(r"(?<=[.!?؟。])\s+|\n+", answer_text or ""):
        for m in answer._CITE_RE.finditer(s):
            if str(n) in [x.strip() for x in m.group(1).split(",")]:
                out.append(answer._MARK_STRIP_RE.sub("", s))
                break
    return " ".join(out) or answer._MARK_STRIP_RE.sub("", answer_text or "")


def _answer_hl(passage: str, answer_text: str, n: int) -> str:
    """The passage sentence the answer's [n] sentences overlap most (numbers count double: "25 MAD", "2 to 4
    days"); "" when nothing overlaps. Ties go to the shorter sentence (the tighter match)."""
    words = search._tokens(_cited_text(answer_text, n))
    want = {t for t in words if not search._is_stop(t)}
    if not want:
        return ""
    best, best_key = "", (0.0, 0.0)
    for s in _SENT_SPLIT.split(passage):
        s = s.strip()
        toks = set(search._tokens(s))
        shared = toks & want
        score = sum(2.0 if any(c.isdigit() for c in t) else 1.0 for t in shared)
        key = (score, score / (len(toks) or 1))
        if score > 0 and key > best_key:
            best, best_key = s, key
    return best


def _hl_map(passages: list, answer_text: str, cited: list[int]) -> dict:
    """{"<n>": {"hl", "hl_at"}} for each cited source whose answer-based sentence was found."""
    out = {}
    for n in cited or []:
        if 1 <= n <= len(passages):
            text = passages[n - 1].text
            hl = _answer_hl(text, answer_text, n)
            if hl:
                out[str(n)] = {"hl": hl, "hl_at": _js_span(text, hl)}
    return out


def _js_span(text: str, part: str) -> list[int] | None:
    """[start, end] of `part` inside `text` as JavaScript string indices (UTF-16 code units), or None."""
    i = text.find(part) if part else -1
    if i < 0:
        return None
    start = len(text[:i].encode("utf-16-le")) // 2
    return [start, start + len(part.encode("utf-16-le")) // 2]


def _left_today(bud, session: str, ip: str) -> int:
    rem = getattr(bud, "remaining", None)
    if callable(rem):
        return int(rem(session, ip))
    with bud._lock:  # read only; budget.py has no public counter per session yet
        bud._reset_if_new_day()
        if bud._spent >= bud.daily_usd:
            return 0
        return max(0, min(bud.session_cap - bud._by_session.get(session, 0), bud.ip_cap - bud._by_ip.get(ip, 0)))


# ---------------------------------------------------------------- app

class _OriginGuard:
    """Pure ASGI (streams pass through untouched): a POST that carries an Origin header outside the CORS list is
    refused before any body is read, model called or cap counted. No Origin header (curl, server-to-server) passes.
    `allowed(origin, path)`: the WordPress demo origins pass on WP_PATHS only."""

    def __init__(self, app, allowed):
        self.app, self.allowed = app, allowed

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("method") == "POST":
            origin = next((v.decode("latin-1") for k, v in scope.get("headers") or [] if k == b"origin"), None)
            if origin is not None and not self.allowed(origin, scope.get("path") or ""):
                await _err("origin_not_allowed", status=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)


class _CORSByPath:
    """Two CORS policies on one app: WP_PATHS also answer the WordPress demo origins (preflight included); every
    other path keeps the site's own list. Pure ASGI, so streams pass through untouched."""

    def __init__(self, app, default: dict, wp: dict):
        self.default, self.wp = CORSMiddleware(app, **default), CORSMiddleware(app, **wp)

    async def __call__(self, scope, receive, send):
        cors = self.wp if scope["type"] == "http" and scope.get("path") in WP_PATHS else self.default
        await cors(scope, receive, send)


def create_app(*, stream=None, translate=None, complete=None, crawl=None, clock=time.time,
               budget: budget_mod.Budget | None = None, quick=None, check_key: bool | None = None,
               visits=None, wp_budget: budget_mod.Budget | None = None) -> FastAPI:
    """`stream`, `translate`, `complete`, `crawl`, `quick`, `visits` are injectable for tests (default: the real
    engine; visits: the store visits.from_env() picks). `quick(messages, schema) -> (data, usage)` makes the
    starter-question call. `budget` guards /api/chat and the other demos; `wp_budget` guards POST /api/answer."""
    app = FastAPI(title="Site chatbot API", docs_url=None, redoc_url=None)
    origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "https://slimanechanchoul.com").split(",") if o.strip()]
    origin_regex = os.environ.get("CORS_ORIGIN_REGEX", r"^http://(localhost|127\.0\.0\.1)(:\d+)?$")
    wp_origins = [o.strip() for o in os.environ.get("WP_DEMO_ORIGINS", "https://playground.wordpress.net").split(",")
                  if o.strip() and o.strip() != "*"]  # never a wildcard: anyone could spend the caps
    cors = dict(allow_origin_regex=origin_regex, allow_methods=["GET", "POST", "OPTIONS"],
                allow_headers=["Content-Type", "Accept"], max_age=7200)
    app.add_middleware(_CORSByPath, default={**cors, "allow_origins": origins},
                       wp={**cors, "allow_origins": origins + [o for o in wp_origins if o not in origins]})
    origin_re = re.compile(origin_regex) if origin_regex else None

    def origin_ok(o: str) -> bool:  # the same test CORSMiddleware makes
        return "*" in origins or o in origins or bool(origin_re is not None and origin_re.fullmatch(o))

    def post_origin_ok(o: str, path: str) -> bool:
        return origin_ok(o) or (path in WP_PATHS and o in wp_origins)

    app.add_middleware(_OriginGuard, allowed=post_origin_ok)

    st = app.state
    st.stream = stream or llm.stream_text
    st.translate = translate  # None -> llm.translate_query when a key is set
    st.complete = complete or llm.complete_json
    st.crawl = crawl or scraper.read_site
    st.quick = quick
    st.real_ai = stream is None
    st.clock = clock
    st.inst = secrets.token_hex(4)
    st.boot = time.monotonic()
    st.budget = budget or budget_mod.Budget(session_cap=_env_int("SESSION_CAP", 12), ip_cap=_env_int("IP_CAP", 30),
                                            daily_usd=_env_float("DAILY_USD", 1.0),
                                            ip_crawl_cap=_env_int("IP_CRAWL_CAP", 5), clock=clock,
                                            ip_upload_cap=_env_int("IP_UPLOAD_CAP", 10))
    # POST /api/answer (WordPress demo) has its own caps, so it can never use up the budget of /ask and /assistant.
    st.wp_budget = wp_budget or budget_mod.Budget(session_cap=_env_int("WP_SESSION_CAP", 12),
                                                  ip_cap=_env_int("WP_IP_CAP", 15),
                                                  daily_usd=_env_float("WP_DAILY_USD", 0.25), clock=clock)
    st.wp_max_tokens = max(100, _env_int("WP_MAX_TOKENS", 500))
    st.restores = RestoreCap(_env_int("IP_RESTORE_CAP", 60), clock=clock)
    st.cache = cache_mod.AnswerCache(clock=clock)
    st.kbs = KBStore(max_entries=_env_int("KB_MAX", 40), ttl_s=_env_int("KB_TTL_S", 3600), clock=clock)
    st.examples = load_examples(st.kbs)
    st.visits = visits if visits is not None else visits_mod.from_env()
    st.hit_caps = visits_mod.HitCaps(_env_int("HIT_IP_CAP", visits_mod.HIT_IP_CAP),
                                     _env_int("HIT_DAILY_CAP", visits_mod.HIT_DAILY_CAP), clock=clock)
    _load_starter_answers(st)
    if check_key is None:
        check_key = st.real_ai and "PYTEST_CURRENT_TEST" not in os.environ
    if check_key and llm.api_key():
        llm.check_key_async()  # background: /healthz says "rejected" (and ai:false) for a bad key

    @app.get("/healthz")
    def healthz():
        key = llm.key_state() if st.real_ai else "ok"
        return {"ok": True, "ai": (llm.available() and key not in ("missing", "rejected")) if st.real_ai else True,
                "kbs": len(st.kbs), "budget": st.budget.snapshot(), "wp_budget": st.wp_budget.snapshot(), "inst": st.inst,
                "up_s": int(time.monotonic() - st.boot), "ai_key": key, "restore": bool(os.environ.get("KB_SECRET"))}

    @app.get("/api/examples")
    def examples():
        out = []
        for ex in st.examples:
            kb: KB = ex["kb"]
            starters = []
            for q in ex["starters"]:
                hit = st.cache.get(kb.source, q)
                item = {"q": q, "cached": hit is not None}
                if hit is not None:
                    item["answer"] = {"text": hit.text, "followups": hit.followups, "cited": hit.cited,
                                      "sources": [{"n": i, "title": p.title, "url": p.url, "passage": p.text}
                                                  for i, p in enumerate(hit.passages, start=1)]}
                starters.append(item)
            out.append({**kb.summary(), "starters": starters})
        return {"examples": out}

    def _starters_quick():
        if st.quick is not None:
            return st.quick
        if st.real_ai and llm.available():
            return lambda messages, schema: llm.quick_json(messages, schema)
        return None

    def _start_starters(label: str, pages: list[Page]) -> dict:
        """Start the starter-question call in a thread; returns a box that gets "qs" (and the cost is recorded)."""
        box: dict = {"done": threading.Event()}
        quick = _starters_quick()
        if quick is None or not st.budget.can_spend():
            box["done"].set()
            return box

        def work():
            try:
                lang = starters_mod.site_language(pages)
                box["lang"] = lang
                box["qs"], usage = starters_mod.ask_model(label, lang, pages, quick)
                st.budget.add_spend(getattr(usage, "cost_usd", 0.0))
            except Exception:  # noqa: BLE001 - templates fill in
                box["qs"] = []
            finally:
                box["done"].set()

        threading.Thread(target=work, daemon=True, name="starters").start()
        return box

    def _finish_starters(kb: KB, box: dict | None, wait_s: float = 3.2) -> None:
        if box is not None:
            box["done"].wait(wait_s)
        try:
            kb.starters = starters_mod.pick(kb.index, kb.pages, kb.label, (box or {}).get("qs"), (box or {}).get("lang"))
        except Exception:  # noqa: BLE001
            kb.starters = []

    def _summary(kb: KB, want_snapshot: bool) -> dict:
        return kb.summary(make_snapshot(kb, st.clock()) if want_snapshot else None)

    # ------------------------------------------------------------ read a site or PDFs

    @app.post("/api/kb")
    async def create_kb(request: Request):
        ip = client_ip(request)
        want_stream = request.query_params.get("stream") in ("1", "true") or \
            "text/event-stream" in request.headers.get("accept", "")
        ctype = request.headers.get("content-type", "")
        if ctype.startswith("multipart/form-data"):
            limit = _env_int("MAX_UPLOAD_MB", 20) * 1_000_000
            if _declared_size(request) > limit + 100_000:
                return _err("too_big", status=413)
            decision = st.budget.check_upload(ip)
            if not decision.allowed:
                return _err(decision.reason, decision.message, status=429)
            st.budget.record_upload(ip)  # before any await, as for crawls: a burst can't all pass the check
            try:  # count the bytes as they arrive: a chunked body has no Content-Length to check first
                form = await _limited_form(request, limit + 100_000)
            except _TooBig:
                return _err("too_big", status=413)
            files, total = [], 0
            for f in form.getlist("files") or form.getlist("file"):
                if not hasattr(f, "read"):
                    continue
                data = await f.read(limit - total + 1)  # never more than what is still allowed (+1 to tell)
                total += len(data)
                if total > limit:
                    return _err("too_big", status=413)
                buf = io.BytesIO(data)
                buf.name = Path(f.filename or "file.pdf").name
                files.append(buf)
            if not files:
                return _err("no_file")
            want_snap = str(form.get("snapshot") or "").lower() in ("1", "true")
            pages, problems = docs.read_pdfs(files)
            if not pages:
                return _err("no_text", "; ".join(problems) or MESSAGES["no_text"], status=422)
            kb = make_kb(pages, ", ".join(b.name for b in files), "pdf", warnings=problems)
            from starlette.concurrency import run_in_threadpool
            await run_in_threadpool(_finish_starters, kb, _start_starters(kb.label, pages))
            st.kbs.put(kb)
            if want_stream:
                return sse_response(iter([sse("done", _summary(kb, want_snap))]))
            return _summary(kb, want_snap)

        if _declared_size(request) > MAX_JSON_BYTES:
            return _err("too_big", status=413)
        try:
            body = await _read_json(request)
            url = body.get("url")
            max_pages = max(1, min(_env_int("MAX_PAGES", 15), int(body.get("max_pages") or 15)))
            want_snap = body.get("snapshot") is True
        except _TooBig:
            return _err("too_big", status=413)
        except Exception:  # noqa: BLE001
            return _err("bad_request")
        try:
            start = scraper.normalize_url(url or "")
        except scraper.CrawlError as e:
            return _err(e.code, e.message)
        decision = st.budget.check_crawl(ip)
        if not decision.allowed:
            return _err(decision.reason, decision.message, status=429)

        early: dict = {}

        def on_kept(page: Page) -> None:  # 5 pages read: start the starter questions during the crawl tail
            early.setdefault("pages", []).append(page)
            if len(early["pages"]) == 5 and "box" not in early:
                early["box"] = _start_starters(_url_label(start), list(early["pages"]))

        def run_crawl(on_page=None):
            if st.crawl is scraper.read_site:
                return st.crawl(start, max_pages, on_page, on_kept=on_kept)
            return st.crawl(start, max_pages, on_page)

        def finish(report) -> KB | None:
            if not report.pages:
                return None
            kb = make_kb(report.pages, _url_label(report.start_url), "site", start_url=report.start_url,
                         warnings=report.warnings)
            kb.rendered = int(getattr(report, "rendered", 0) or 0)
            box = early.get("box") or _start_starters(kb.label, kb.pages)
            _finish_starters(kb, box)
            st.kbs.put(kb)
            return kb

        st.budget.record_crawl(ip)  # right after the check, before any await: a burst can't all pass the check
        if not want_stream:
            from starlette.concurrency import run_in_threadpool
            try:
                report = await run_in_threadpool(run_crawl)
            except scraper.CrawlError as e:
                return _err(e.code, e.message, status=422)
            kb = await run_in_threadpool(finish, report)
            if kb is None:
                code = "js_only" if report.js_only else "no_text"
                return _err(code, (report.warnings or [scraper.MESSAGES[code]])[0], status=422)
            return _summary(kb, want_snap)

        def events() -> Iterator[str]:
            q: queue.Queue = queue.Queue()
            box: dict = {}

            def work():
                try:
                    box["report"] = run_crawl(lambda n, u: q.put(("progress", {"n": n, "url": u})))
                except scraper.CrawlError as e:
                    box["error"] = {"code": e.code, "message": e.message}
                except Exception:  # noqa: BLE001
                    box["error"] = {"code": "server_error", "message": MESSAGES["server_error"]}
                q.put(None)

            threading.Thread(target=work, daemon=True).start()
            yield sse("start", {"url": start, "max_pages": max_pages})
            while (item := q.get()) is not None:
                yield sse(*item)
            if "error" in box:
                yield sse("error", box["error"])
                return
            report = box["report"]
            kb = finish(report)
            if kb is None:
                code = "js_only" if report.js_only else "no_text"
                yield sse("error", {"code": code, "message": (report.warnings or [scraper.MESSAGES[code]])[0]})
                return
            yield sse("done", _summary(kb, want_snap))

        return sse_response(events())

    def _kb_or_restore(kb_id: str, snap, ip: str) -> tuple[KB | None, str | None]:
        """(kb, None) or (None, error code). A missing KB is rebuilt from a valid signed snapshot (no crawl, no AI)."""
        kb = st.kbs.get(kb_id)
        if kb is not None or not snap:
            return kb, (None if kb is not None else "kb_expired")
        pages = verify_snapshot(snap, kb_id, st.clock())
        if pages is None:
            return None, "kb_expired"
        if not st.restores.take(ip):
            return None, "ip_restore_cap"
        kb = make_kb(pages, str(snap.get("label") or ""), str(snap.get("kind") or "site"), kb_id=kb_id,
                     start_url=str(snap.get("url") or ""))
        kb.rendered = sum(1 for p in pages if p.rendered)
        st.kbs.put(kb)
        return kb, None

    # ------------------------------------------------------------ chat

    def _sse_error(code: str) -> StreamingResponse:
        return sse_response(iter([sse("error", {"code": code, "message": MESSAGES.get(code, MESSAGES["bad_request"])})]))

    @app.post("/api/chat")
    async def chat(request: Request):
        t0 = time.monotonic()
        ip = client_ip(request)
        try:
            body = await _read_json(request)
            msgs = [{"role": m["role"], "content": m["content"][:MAX_HISTORY_CHARS]} for m in body.get("messages") or []
                    if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
            kb_id = str(body.get("kb_id") or "")
            session = str(body.get("session_id") or "")[:64] or f"ip:{ip}"
        except _TooBig:
            return _sse_error("too_big")
        except Exception:  # noqa: BLE001
            return _sse_error("bad_request")
        if not msgs or msgs[-1]["role"] != "user" or not msgs[-1]["content"].strip():
            return _sse_error("bad_request")
        from starlette.concurrency import run_in_threadpool
        kb, code = await run_in_threadpool(_kb_or_restore, kb_id, body.get("restore"), ip)
        if kb is None:
            return _sse_error(code or "kb_expired")
        question = msgs[-1]["content"].strip()[:MAX_QUESTION_CHARS]
        history = msgs[:-1][-MAX_HISTORY:]
        return sse_response(_chat_events(st, kb, question, history, session, ip, t0))

    # ------------------------------------------------------------ extract + exports

    @app.post("/api/extract")
    async def extract_endpoint(request: Request):
        ip = client_ip(request)
        try:
            body = await _read_json(request)
        except _TooBig:
            return _err("too_big", status=413)
        except Exception:  # noqa: BLE001
            return _err("bad_request")
        from starlette.concurrency import run_in_threadpool
        kb, code = await run_in_threadpool(_kb_or_restore, str(body.get("kb_id") or ""), body.get("restore"), ip)
        if kb is None:
            return _err(code or "kb_expired", status=429 if code == "ip_restore_cap" else 404)
        raw = body.get("fields")
        fields = extract.parse_fields(", ".join(raw) if isinstance(raw, list) else str(raw or ""))
        if not fields:
            return _err("no_fields")
        if st.complete is llm.complete_json and not llm.available():
            return _err("no_ai", status=503)
        session = str(body.get("session_id") or "")[:64] or f"ip:{ip}"
        return await run_in_threadpool(_extract, st, kb, fields, session, ip)

    @app.get("/api/kb/{name}")
    def export(name: str):
        kb_id, _, ext = name.rpartition(".")
        kb = st.kbs.get(kb_id) if ext in ("csv", "json") else None
        if kb is None:
            return _err("kb_expired", status=404)
        if ext == "json":
            return {**kb.summary(), "pages_text": [{"url": p.url, "title": p.title, "words": p.words, "text": p.text}
                                                   for p in kb.pages], "fields": kb.fields, "rows": kb.rows}
        out = io.StringIO()
        if kb.rows:
            cols = kb.fields + ["source_url", "source_title"]
            w = csv.writer(out)
            w.writerow(cols)
            for r in kb.rows:
                w.writerow([_csv_safe(r.get(c, "")) for c in kb.fields] + [r["source"]["url"], _csv_safe(r["source"]["title"])])
        else:
            w = csv.writer(out)
            w.writerow(["url", "title", "words"])
            for p in kb.pages:
                w.writerow([p.url, _csv_safe(p.title), p.words])
        return Response("﻿" + out.getvalue(), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{kb.id}.csv"'})

    # ------------------------------------------------------------ answers from given passages (WordPress plugin demo)

    @app.post("/api/answer")
    async def given_answer(request: Request):
        ip = client_ip(request)
        if _declared_size(request) > WP_MAX_BYTES:
            return _err("too_big", status=413)
        raw = b""
        async for chunk in request.stream():  # counted as it arrives: a chunked body has no Content-Length
            raw += chunk
            if len(raw) > WP_MAX_BYTES:
                return _err("too_big", status=413)
        try:
            req = _wp_request(json.loads(raw.decode("utf-8") or "null"))
        except _BadRequest as e:
            return _err("bad_request", str(e))
        except (ValueError, RecursionError):
            return _err("bad_request")
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(_wp_answer, st, req, ip)

    # ------------------------------------------------------------ visit log (slimanechanchoul.com/hit.js)

    @app.post("/api/hit")
    async def hit(request: Request):
        origin = request.headers.get("origin")
        if not origin or not origin_ok(origin):  # a foreign Origin is already refused by _OriginGuard
            return _err("origin_not_allowed", status=403)
        if _declared_size(request) > visits_mod.MAX_LEAVE_BODY:
            return _err("too_big", status=413)
        raw = b""
        async for chunk in request.stream():
            raw += chunk
            if len(raw) > visits_mod.MAX_LEAVE_BODY:
                return _err("too_big", status=413)
        body = _hit_json(raw)
        leave = visits_mod.is_leave(body)
        if len(raw) > visits_mod.MAX_BODY and not leave:  # a page view keeps its 2 KB; only a summary may be longer
            return _err("too_big", status=413)
        ua = request.headers.get("user-agent", "")
        if visits_mod.is_bot(ua):
            return Response(status_code=204, headers=NO_STORE)  # dropped quietly, not counted
        if body is _NOT_JSON:
            return _err("bad_request")
        if leave:  # time on page and clicks, at most one per page view (hit.js); no IP, no geo, no device
            event = visits_mod.make_leave(body, st.clock())
        else:  # a page view, exactly as before the leave summaries (bodies without "type" included)
            event = visits_mod.make_event(body, request.headers, ua, st.clock())
        if event is None:
            return _err("bad_request")
        if not st.hit_caps.take(client_ip(request)):  # the IP is used here only, never stored
            return _err("hit_cap", status=429)
        return Response(status_code=204, headers=NO_STORE,
                        background=BackgroundTask(visits_mod.safe_add, st.visits, event))

    @app.get("/api/stats")
    def visit_stats(request: Request):
        key = os.environ.get("STATS_KEY", "").strip()
        if not key:
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        given = request.headers.get("x-stats-key", "")
        if not hmac.compare_digest(given.encode("utf-8"), key.encode("utf-8")):
            return _err("unauthorized", status=401)
        q = request.query_params
        out = visits_mod.stats(st.visits, st.clock(), _int_param(q.get("days"), 7), _int_param(q.get("limit"), 200))
        if out is None:
            return _err("store_unavailable", status=503)
        return JSONResponse(out, headers=NO_STORE)

    return app


NO_STORE = {"Cache-Control": "no-store"}
_NOT_JSON = object()


def _hit_json(raw: bytes):
    """A /api/hit body as JSON, or _NOT_JSON (bad UTF-8, not JSON, or nested too deep to parse)."""
    try:
        return json.loads(raw.decode("utf-8") or "null")
    except (ValueError, RecursionError):
        return _NOT_JSON


def _int_param(v: str | None, default: int) -> int:
    try:
        return int(v) if v is not None else default
    except ValueError:
        return default


def _csv_safe(v) -> str:
    s = str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@") else s  # no formula injection in Excel


def _load_starter_answers(st) -> None:
    """Optional warm cache: STARTER_ANSWERS=path to [{"kb_id", "question", "text", "followups", "cited"}]."""
    path = os.environ.get("STARTER_ANSWERS") or str(Path(__file__).with_name("starter_answers.json"))
    if not Path(path).exists():
        return
    try:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    for r in rows:
        kb = st.kbs.get(r.get("kb_id", ""))
        if kb is None:
            continue
        found = kb.index.search(r["question"])
        passages = [search.trim_passage(p, found.search_query) for p in found.passages[:answer.MAX_PASSAGES]]
        st.cache.put(kb.source, r["question"], answer.AnswerResult(
            r["text"], "answered", passages, r.get("cited", []), r.get("followups", []), llm.Usage()))


class _Trimmed:
    """A SearchResult whose passages are already trimmed, so the sources event and the model see the same text."""

    def __init__(self, found, passages):
        self.matched, self.search_query, self.lang, self.passages = found.matched, found.search_query, found.lang, passages


def _chat_events(st, kb: KB, question: str, history: list[dict], session: str, ip: str, t0: float) -> Iterator[str]:
    bud = st.budget
    try:
        prev = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
        translate = st.translate
        if translate is None and llm.available():
            def translate(text: str, lang: str, hint: str = "") -> str:
                if not bud.check(session, ip).allowed:
                    return text
                return llm.translate_query(text, lang, hint=hint, on_usage=lambda u: bud.record(
                    session, ip, llm.Usage(u.prompt_tokens, u.completion_tokens, u.cost_usd, u.model, 0)))
        found = kb.index.search(question, prev_question=prev, translate=translate)
        passages = [search.trim_passage(p, found.search_query) for p in found.passages[:answer.MAX_PASSAGES]]
        lang = found.lang
        yield sse("meta", {"lang": lang, "dir": "rtl" if lang == "ar" else "ltr", "pages": len(kb.pages)})
        # The highlight guess uses this question's own words (a follow-up's earlier question would pull it to the
        # wrong sentence); a cached answer is known already, so its sentences pick the highlight at once.
        own = question if search._content(search._tokens(question)) else found.search_query
        any_turn = kb.kind == "example"  # pinned examples: an exact follow-up chip is cached even mid-chat
        pre = st.cache.get(kb.source, question) if (not history or any_turn) and passages else None
        pre_hl = _hl_map(passages, pre.text, pre.cited) if pre is not None else {}
        sources = []
        for n, p in enumerate(passages, start=1):
            hl, terms = _highlight(p, own)
            hl = pre_hl.get(str(n), {}).get("hl") or hl
            sources.append({"n": n, "title": p.title, "url": p.url, "passage": p.text, "hl": hl,
                            "hl_at": _js_span(p.text, hl), "terms": terms})
        yield sse("sources", sources)

        run = answer.ask(kb.label, question, _Trimmed(found, passages), history, source=kb.source, session=session, ip=ip,
                         cache=st.cache, budget=bud, stream=st.stream, trim=lambda p, q: p, cache_any_turn=any_turn)
        ttfw = None
        for chunk in run:
            if ttfw is None:
                ttfw = int((time.monotonic() - t0) * 1000)
                res = run.result  # set before the text for no_match / cached / extractive / early error
                if res is not None and res.status in ("extractive", "error") and res.note:
                    code = _REASON_BY_MESSAGE.get(res.note, "ai_busy")
                    if code == "ai_busy" and st.real_ai and llm.key_state() in ("missing", "rejected"):
                        code = "ai_unavailable"
                    yield sse("notice", {"code": code, "message": res.note if code not in ("ai_busy", "ai_unavailable")
                                         else MESSAGES[code]})
            if chunk:
                yield sse("delta", {"text": chunk})
        res = run.result
        yield sse("followups", _answerable(kb, res.followups, question) if res else [])
        yield sse("done", {"ttfw_ms": ttfw if ttfw is not None else int((time.monotonic() - t0) * 1000),
                           "cached": bool(res and res.status == "cached"),
                           "cost": round(res.usage.cost_usd, 6) if res else 0.0,
                           "left_today": _left_today(bud, session, ip),
                           "status": res.status if res else "error", "cited": res.cited if res else [],
                           "model": res.usage.model if res else "",
                           # the sentence each cited source really backs, picked from the answer: the page
                           # moves the source card's highlight to it ({"<n>": {hl, hl_at}})
                           "hl": _hl_map(passages, res.text, res.cited) if res and res.status in ("answered", "cached")
                           else {},
                           "closest": _closest(kb, question, res)})
    except Exception:  # noqa: BLE001 - never a stack trace to the visitor
        import logging
        logging.getLogger(__name__).exception("chat failed")
        yield sse("error", {"code": "server_error", "message": MESSAGES["server_error"]})


_NOT_FOUND = re.compile(
    r"couldn't find|could not find|can't find|cannot find|not (?:mentioned|covered|listed|available) (?:on|in)|"
    r"no information|doesn't (?:say|mention)|does not (?:say|mention)|je n'ai pas trouvé|n'ai pas pu trouver|"
    r"je ne trouve pas|ne trouve pas|pas d'information|ne (?:mentionne|précise|contient) pas|"
    r"لم أتمكن من العثور|لم أجد|لا تتوفر|لا توجد معلومات|لا يذكر|لم يتم ذكر|غير متوفرة", re.I)


def _closest(kb: KB, question: str, res) -> dict | None:
    """{title, url} of the nearest page when the site does not answer the question (no_match, or an answer that
    says it could not find it); None otherwise."""
    if res is None:
        return None
    if not (res.status == "no_match" or (res.status in ("answered", "cached") and _NOT_FOUND.search(res.text[:400]))):
        return None
    try:
        p = kb.index.closest(question)
    except Exception:  # noqa: BLE001
        return None
    if p is None or not p.url.startswith(("http://", "https://")):
        return None
    return {"title": p.title, "url": p.url}


def _answerable(kb: KB, followups: list[str], question: str) -> list[str]:
    """Only the suggested follow-ups the site can answer: a chip must never lead to "not on this site".
    Local search, no model call; a follow-up in another language than the site is kept (it would need
    a translation call to check)."""
    out = []
    for f in followups or []:
        try:
            if kb.index._foreign(f) or kb.index.search(f, prev_question=question).matched:
                out.append(f)
        except Exception:  # noqa: BLE001 - a check that fails keeps the chip
            out.append(f)
    return out


class _BadRequest(ValueError):
    """A /api/answer body that breaks a rule; the message says which (visitor-safe, no internals)."""


def _text(v, name: str, max_chars: int, *, required: bool = True) -> str:
    if v is None and not required:
        return ""
    if not isinstance(v, str) or (required and not v.strip()):
        raise _BadRequest(f"{name} must be a non-empty text.")
    if len(v) > max_chars:
        raise _BadRequest(f"{name} is longer than {max_chars} characters.")
    return v.strip()


def _wp_request(body) -> dict:
    """Validate a /api/answer body; raises _BadRequest."""
    if not isinstance(body, dict):
        raise _BadRequest("Send a JSON object.")
    translate = body.get("translate_query", False)
    if not isinstance(translate, bool):
        raise _BadRequest("translate_query must be true or false.")
    question = _text(body.get("question"), "question", WP_QUESTION_CHARS)
    lang = _text(body.get("site_lang"), "site_lang", 10, required=translate)
    if lang and not re.fullmatch(r"[a-z]{2}(?:[-_][A-Za-z]{2})?", lang):
        raise _BadRequest("site_lang must be a language code such as en or fr.")
    label = " ".join(_text(body.get("site_label"), "site_label", WP_LABEL_CHARS, required=False).split())
    session = _text(body.get("session_id"), "session_id", 64, required=False)
    # Words, not sentences: an instruction written in the hint reaches the model only as a list of loose words.
    hint = ", ".join(re.findall(r"[\w'-]{2,30}", _text(body.get("hint"), "hint", 900, required=False))[:WP_HINT_WORDS])
    history = body.get("history") or []
    if not isinstance(history, list) or len(history) > WP_HISTORY_MESSAGES:
        raise _BadRequest("history holds at most 3 turns.")
    turns = []
    for m in history:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            raise _BadRequest("Each history item needs a role: user or assistant.")
        turns.append({"role": m["role"], "content": _text(m.get("content"), "history content", WP_HISTORY_CHARS)})
    passages = []
    if not translate:
        raw = body.get("passages")
        if not isinstance(raw, list) or not 1 <= len(raw) <= WP_MAX_PASSAGES:
            raise _BadRequest(f"Send 1 to {WP_MAX_PASSAGES} passages.")
        for p in raw:
            if not isinstance(p, dict):
                raise _BadRequest("Each passage needs url, title and text.")
            url = _text(p.get("url"), "passage url", 2000)
            if not re.match(r"^https?://[^\s/]+", url, re.I):
                raise _BadRequest("Passage URLs must start with http:// or https://.")
            title = " ".join(_text(p.get("title"), "passage title", 300).split())
            passages.append(search.Passage(url, title, _text(p.get("text"), "passage text", WP_PASSAGE_CHARS)))
    return {"question": question, "lang": lang, "label": label or "this site", "session": session,
            "history": turns, "passages": passages, "translate": translate, "hint": hint}


class _Given:
    """A SearchResult made of the passages the site sent: matched, in the site's own order."""

    def __init__(self, question: str, passages: list):
        self.matched, self.search_query, self.passages = True, question, passages
        self.lang = search.detect_lang(question)


def _wp_translate(st, req: dict, session: str, ip: str) -> str:
    """translate_query mode: one short model call, counted toward the same caps as an answer (on its own, this route
    would otherwise be a free text tool). The question comes back unchanged when a cap is reached or the call fails."""
    bud = st.wp_budget
    translate = st.translate
    usages: list[llm.Usage] = []
    if translate is None and llm.available():
        def translate(text: str, lang: str, hint: str = "") -> str:
            return llm.translate_query(text, lang, hint=hint, on_usage=usages.append)
    if translate is None or not bud.check(session, ip, reserve=True).allowed:
        return req["question"]
    query = req["question"]
    try:
        try:  # the site's own words make the translated search find them ("refund", not "reimbursement")
            out = translate(req["question"], req["lang"], hint=req["hint"]) if req["hint"] else \
                translate(req["question"], req["lang"])
        except TypeError:  # a translate(text, lang) without the hint
            out = translate(req["question"], req["lang"])
        query = (out or "").strip() or req["question"]
    except Exception:  # noqa: BLE001 - a failed translation keeps the question
        query = req["question"]
    finally:
        u = usages[-1] if usages else llm.Usage()
        bud.record(session, ip, llm.Usage(u.prompt_tokens, u.completion_tokens, u.cost_usd, u.model, 1), reserved=True)
    return query


_SENTENCE = re.compile(r"[^.!?\u061F\n]+[.!?\u061F]*")
_CITE_MARK = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")


def _uncite_not_found(text: str) -> str:
    """A "could not find" sentence cites nothing: drop the [n] marks of the sentences that say so, so the page the
    model named is not shown as a source. The other sentences of a partial answer keep theirs."""
    return _SENTENCE.sub(lambda m: _CITE_MARK.sub("", m.group(0)) if _NOT_FOUND.search(m.group(0)) else m.group(0), text)


def _wp_source(req: dict) -> str:
    """Cache key of everything in the prompt but the question: site name, site language and each passage's url, title
    and text (the same passages under another site name must not get that site's cached answer)."""
    blob = json.dumps([req["label"], req["lang"], [(p.url, p.title, p.text) for p in req["passages"]]],
                      ensure_ascii=False)
    return "wp:" + hashlib.sha1(blob.encode("utf-8", "replace")).hexdigest()[:16]


def _wp_answer(st, req: dict, ip: str) -> JSONResponse:
    bud = st.wp_budget
    session = req["session"] or f"ip:{ip}"
    if req["translate"]:
        query = _wp_translate(st, req, session, ip)
        return JSONResponse({"query": query[:WP_QUESTION_CHARS]}, headers=NO_STORE)

    passages = req["passages"]
    run = answer.ask(req["label"], req["question"], _Given(req["question"], passages), req["history"],
                     source=_wp_source(req), session=session, ip=ip, cache=st.cache, budget=bud,
                     stream=st.stream, trim=lambda p, q: p, max_passages=len(passages),
                     system=answer.GIVEN_PASSAGES_SYSTEM, max_tokens=st.wp_max_tokens)
    try:
        for _ in run:
            pass
        res = run.result
    except Exception:  # noqa: BLE001 - never a stack trace to the visitor
        import logging
        logging.getLogger(__name__).exception("answer failed")
        return _err("server_error", status=500)
    sources = [{"n": i, "title": p.title, "url": p.url, "passage": p.text} for i, p in enumerate(passages, start=1)]
    if res is None or res.status not in ("answered", "cached"):
        note = res.note if res is not None else ""
        code = _REASON_BY_MESSAGE.get(note, "ai_busy")
        if code == "ai_busy" and st.real_ai and llm.key_state() in ("missing", "rejected"):
            code = "ai_unavailable"
        message = note if code not in ("ai_busy", "ai_unavailable") else MESSAGES[code]
        return JSONResponse({"answer": message, "sources": sources, "mode": "extractive", "not_found": False,
                             "cited": [], "notice": code}, headers=NO_STORE)
    text = res.text
    not_found = bool(_NOT_FOUND.search(text[:400]))
    if not_found:
        text = _uncite_not_found(text)
    cited = [n for n in answer.parse_citations(text, len(sources)) if 1 <= n <= len(sources)]
    used = [sources[n - 1] for n in cited] if cited else ([] if not_found else sources)
    return JSONResponse({"answer": text, "sources": used, "mode": "ai", "not_found": not_found, "cited": cited,
                         "notice": None}, headers=NO_STORE)


EXTRACT_STOP = {  # budget stops for extraction: no "passages instead", Extract shows none
    "session_cap": "You have used the AI requests for this visit, so extraction stopped here.",
    "ip_cap": "Today's AI requests for your network are used up, so extraction stopped here. Try again tomorrow.",
    "global_budget": "The demo has used today's AI budget, so extraction stopped here. Try again tomorrow.",
}


def _extract(st, kb: KB, fields: list[tuple[str, str]], session: str, ip: str):
    bud = st.budget
    rows, warnings, read = [], [], 0
    for p in kb.pages[:_env_int("EXTRACT_MAX_PAGES", 5)]:
        decision = bud.check(session, ip, reserve=True)
        if not decision.allowed:
            warnings.append(EXTRACT_STOP.get(decision.reason, decision.message))
            break
        used: dict = {}

        def complete(messages, schema, name="result"):
            data, model = st.complete(messages, schema, name)
            pt = sum(len(m["content"]) for m in messages) // 4
            ct = len(json.dumps(data)) // 4
            used["usage"] = llm.Usage(pt, ct, llm.estimate_cost(model, pt, ct), model, 1)
            return data, model

        try:
            found = extract.extract_rows(p.url, p.text, fields, complete=complete)
        except llm.LLMError:
            warnings.append(MESSAGES["extract_busy"])
            break
        finally:  # settles the reservation (a failed call counts nothing)
            bud.record(session, ip, used.get("usage") or llm.Usage(), reserved=True)
        read += 1
        rows += [{**r, "source": {"url": p.url, "title": p.title}} for r in found]
    if len(kb.pages) > read and not warnings:
        warnings.append(f"Read the first {read} of {len(kb.pages)} pages (demo limit).")
    kb.fields, kb.rows = [label for _, label in fields], rows
    return {"kb_id": kb.id, "fields": kb.fields, "rows": rows, "pages_read": read, "warnings": warnings}


app = create_app()
