"""HTTP API for the two front ends (Demo 1 "Ask any website", Demo 2 widget/studio).

Run from the repo root:  uvicorn api.main:app --port 8000

Endpoints
  GET  /healthz                    -> {ok, ai, kbs, budget}
  GET  /api/examples               -> pre-indexed example sources (eval/fixtures) + starter questions,
                                      with the cached answer of each starter when the shared cache has one
  POST /api/kb                     -> JSON {url, max_pages?} or multipart PDFs (field "files")
                                      -> {kb_id, label, kind, url, lang, pages, words, warnings,
                                          page_list:[{title, url}] (first 20 pages, crawl order)}
       Daily per-IP caps: site reads IP_CRAWL_CAP (429 error code ip_crawl_cap), PDF uploads IP_UPLOAD_CAP
       (429 error code ip_upload_cap).
       Crawl progress: send the same POST with ?stream=1 (or Accept: text/event-stream). The answer is then
       an SSE stream: progress{n, url} (one per page read) ... then done{<the JSON above>} or error{code, message}.
  POST /api/chat {kb_id, messages:[{role, content}], session_id}
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
          global_budget, ai_busy.
  POST /api/extract {kb_id, fields}  -> {kb_id, fields, rows:[{<label>: value, ..., source:{url, title}}],
                                         pages_read, warnings} (or {error:{code, message}})
       Multipart uploads are counted while they stream: over MAX_UPLOAD_MB -> 413 too_big even without a
       Content-Length header.
  GET  /api/kb/{id}.json           -> pages (text only) + the last extracted rows
  GET  /api/kb/{id}.csv            -> the last extracted rows as CSV (or the page list when nothing was extracted)

Env: CORS_ORIGINS (comma list, default https://slimanechanchoul.com), CORS_ORIGIN_REGEX (default any
http://localhost:* / 127.0.0.1:*), SESSION_CAP, IP_CAP, DAILY_USD, IP_CRAWL_CAP, IP_UPLOAD_CAP (10 PDF uploads per IP per day), KB_TTL_S (3600),
KB_MAX (40), MAX_PAGES (15), EXTRACT_MAX_PAGES (5), MAX_UPLOAD_MB (20), STARTER_ANSWERS (json file),
TRUSTED_PROXY_HOPS (1: the per-IP caps use the X-Forwarded-For entry our proxy added, never the visitor's own),
CLIENT_IP_HEADER (optional platform header with the visitor address), OPENROUTER_API_KEY.
"""
from __future__ import annotations

import csv
import io
import json
import os
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

import answer  # noqa: E402
import budget as budget_mod  # noqa: E402
import cache as cache_mod  # noqa: E402
import docs  # noqa: E402
import extract  # noqa: E402
import llm  # noqa: E402
import scraper  # noqa: E402
import search  # noqa: E402
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

MESSAGES = {
    "kb_expired": "This site was read more than an hour ago and has been cleared. Reading it again now.",
    "bad_request": "Something was missing in that request. Please try again.",
    "server_error": "Something went wrong on our side. Please try again in a minute.",
    "ai_busy": "The AI is busy right now, so here are the most relevant passages from the site instead.",
    "extract_busy": "The AI is busy right now, so extraction stopped early. Please try again in a minute.",
    "no_file": "Add a PDF file or a website address.",
    "too_big": "That upload is too big for the demo.",
    "no_text": scraper.MESSAGES["no_text"],
    "no_fields": "Name the fields to extract, separated by commas (for example: product, price).",
    "no_ai": "Extraction needs the AI, which is not available right now.",
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

    @property
    def words(self) -> int:
        return sum(p.words for p in self.pages)

    def summary(self) -> dict:
        return {"kb_id": self.id, "label": self.label, "kind": self.kind, "url": self.start_url,
                "lang": self.index.lang, "pages": len(self.pages), "words": self.words, "warnings": self.warnings,
                "page_list": [{"title": p.title, "url": p.url} for p in self.pages[:PAGE_LIST_MAX]]}


def make_kb(pages: list[Page], label: str, kind: str, *, kb_id: str | None = None, start_url: str = "",
            warnings: list[str] | None = None) -> KB:
    return KB(kb_id or uuid.uuid4().hex[:12], label, kind, pages, search.Index(pages), cache_mod.source_id(pages),
              start_url, list(warnings or []))


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

def create_app(*, stream=None, translate=None, complete=None, crawl=None, clock=time.time,
               budget: budget_mod.Budget | None = None) -> FastAPI:
    """`stream`, `translate`, `complete`, `crawl` are injectable for tests (default: the real engine)."""
    app = FastAPI(title="Site chatbot API", docs_url=None, redoc_url=None)
    origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "https://slimanechanchoul.com").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins,
                       allow_origin_regex=os.environ.get("CORS_ORIGIN_REGEX", r"^http://(localhost|127\.0\.0\.1)(:\d+)?$"),
                       allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Accept"])

    st = app.state
    st.stream = stream or llm.stream_text
    st.translate = translate  # None -> llm.translate_query when a key is set
    st.complete = complete or llm.complete_json
    st.crawl = crawl or scraper.read_site
    st.budget = budget or budget_mod.Budget(session_cap=_env_int("SESSION_CAP", 12), ip_cap=_env_int("IP_CAP", 30),
                                            daily_usd=_env_float("DAILY_USD", 1.0),
                                            ip_crawl_cap=_env_int("IP_CRAWL_CAP", 5), clock=clock,
                                            ip_upload_cap=_env_int("IP_UPLOAD_CAP", 10))
    st.cache = cache_mod.AnswerCache(clock=clock)
    st.kbs = KBStore(max_entries=_env_int("KB_MAX", 40), ttl_s=_env_int("KB_TTL_S", 3600), clock=clock)
    st.examples = load_examples(st.kbs)
    _load_starter_answers(st)

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "ai": llm.available() or stream is not None, "kbs": len(st.kbs), "budget": st.budget.snapshot()}

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
            pages, problems = docs.read_pdfs(files)
            if not pages:
                return _err("no_text", "; ".join(problems) or MESSAGES["no_text"], status=422)
            kb = make_kb(pages, ", ".join(b.name for b in files), "pdf", warnings=problems)
            st.kbs.put(kb)
            if want_stream:
                return sse_response(iter([sse("done", kb.summary())]))
            return kb.summary()

        if _declared_size(request) > MAX_JSON_BYTES:
            return _err("too_big", status=413)
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError
            url = body.get("url")
            max_pages = max(1, min(_env_int("MAX_PAGES", 15), int(body.get("max_pages") or 15)))
        except Exception:  # noqa: BLE001
            return _err("bad_request")
        try:
            start = scraper.normalize_url(url or "")
        except scraper.CrawlError as e:
            return _err(e.code, e.message)
        decision = st.budget.check_crawl(ip)
        if not decision.allowed:
            return _err(decision.reason, decision.message, status=429)

        def finish(report) -> KB | None:
            if not report.pages:
                return None
            kb = make_kb(report.pages, _url_label(report.start_url), "site", start_url=report.start_url,
                         warnings=report.warnings)
            st.kbs.put(kb)
            return kb

        st.budget.record_crawl(ip)  # right after the check, before any await: a burst can't all pass the check
        if not want_stream:
            try:
                from starlette.concurrency import run_in_threadpool
                report = await run_in_threadpool(st.crawl, start, max_pages)
            except scraper.CrawlError as e:
                return _err(e.code, e.message, status=422)
            kb = finish(report)
            if kb is None:
                code = "js_only" if report.js_only else "no_text"
                return _err(code, (report.warnings or [scraper.MESSAGES[code]])[0], status=422)
            return kb.summary()

        def events() -> Iterator[str]:
            q: queue.Queue = queue.Queue()
            box: dict = {}

            def work():
                try:
                    box["report"] = st.crawl(start, max_pages, lambda n, u: q.put(("progress", {"n": n, "url": u})))
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
            yield sse("done", kb.summary())

        return sse_response(events())

    # ------------------------------------------------------------ chat

    @app.post("/api/chat")
    async def chat(request: Request):
        t0 = time.monotonic()
        ip = client_ip(request)
        if _declared_size(request) > MAX_JSON_BYTES:
            return sse_response(iter([sse("error", {"code": "bad_request", "message": MESSAGES["bad_request"]})]))
        try:
            body = await request.json()
            msgs = [{"role": m["role"], "content": m["content"][:MAX_HISTORY_CHARS]} for m in body.get("messages") or []
                    if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
            kb_id = str(body.get("kb_id") or "")
            session = str(body.get("session_id") or "")[:64] or f"ip:{ip}"
        except Exception:  # noqa: BLE001
            return sse_response(iter([sse("error", {"code": "bad_request", "message": MESSAGES["bad_request"]})]))
        if not msgs or msgs[-1]["role"] != "user" or not msgs[-1]["content"].strip():
            return sse_response(iter([sse("error", {"code": "bad_request", "message": MESSAGES["bad_request"]})]))
        kb = st.kbs.get(kb_id)
        if kb is None:
            return sse_response(iter([sse("error", {"code": "kb_expired", "message": MESSAGES["kb_expired"]})]))
        question = msgs[-1]["content"].strip()[:MAX_QUESTION_CHARS]
        history = msgs[:-1][-MAX_HISTORY:]
        return sse_response(_chat_events(st, kb, question, history, session, ip, t0))

    # ------------------------------------------------------------ extract + exports

    @app.post("/api/extract")
    async def extract_endpoint(request: Request):
        ip = client_ip(request)
        if _declared_size(request) > MAX_JSON_BYTES:
            return _err("too_big", status=413)
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError
        except Exception:  # noqa: BLE001
            return _err("bad_request")
        kb = st.kbs.get(str(body.get("kb_id") or ""))
        if kb is None:
            return _err("kb_expired", status=404)
        raw = body.get("fields")
        fields = extract.parse_fields(", ".join(raw) if isinstance(raw, list) else str(raw or ""))
        if not fields:
            return _err("no_fields")
        if st.complete is llm.complete_json and not llm.available():
            return _err("no_ai", status=503)
        session = str(body.get("session_id") or "")[:64] or f"ip:{ip}"
        from starlette.concurrency import run_in_threadpool
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

    return app


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
            def translate(text: str, lang: str) -> str:
                if not bud.check(session, ip).allowed:
                    return text
                return llm.translate_query(text, lang, on_usage=lambda u: bud.record(
                    session, ip, llm.Usage(u.prompt_tokens, u.completion_tokens, u.cost_usd, u.model, 0)))
        found = kb.index.search(question, prev_question=prev, translate=translate)
        passages = [search.trim_passage(p, found.search_query) for p in found.passages[:answer.MAX_PASSAGES]]
        lang = found.lang
        yield sse("meta", {"lang": lang, "dir": "rtl" if lang == "ar" else "ltr", "pages": len(kb.pages)})
        # The highlight guess uses this question's own words (a follow-up's earlier question would pull it to the
        # wrong sentence); a cached answer is known already, so its sentences pick the highlight at once.
        own = question if search._content(search._tokens(question)) else found.search_query
        pre = st.cache.get(kb.source, question) if not history and passages else None
        pre_hl = _hl_map(passages, pre.text, pre.cited) if pre is not None else {}
        sources = []
        for n, p in enumerate(passages, start=1):
            hl, terms = _highlight(p, own)
            hl = pre_hl.get(str(n), {}).get("hl") or hl
            sources.append({"n": n, "title": p.title, "url": p.url, "passage": p.text, "hl": hl,
                            "hl_at": _js_span(p.text, hl), "terms": terms})
        yield sse("sources", sources)

        run = answer.ask(kb.label, question, _Trimmed(found, passages), history, source=kb.source, session=session, ip=ip,
                         cache=st.cache, budget=bud, stream=st.stream, trim=lambda p, q: p)
        ttfw = None
        for chunk in run:
            if ttfw is None:
                ttfw = int((time.monotonic() - t0) * 1000)
                res = run.result  # set before the text for no_match / cached / extractive / early error
                if res is not None and res.status in ("extractive", "error") and res.note:
                    code = _REASON_BY_MESSAGE.get(res.note, "ai_busy")
                    yield sse("notice", {"code": code, "message": res.note if code != "ai_busy" else MESSAGES["ai_busy"]})
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
                           else {}})
    except Exception:  # noqa: BLE001 - never a stack trace to the visitor
        import logging
        logging.getLogger(__name__).exception("chat failed")
        yield sse("error", {"code": "server_error", "message": MESSAGES["server_error"]})


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
