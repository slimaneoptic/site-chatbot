# Upgrade contract: wave 1 (branch `upgrade-v2`, 2026-10-13)

Source plan: `04-demos/site-chatbot/design/UPGRADE-PLAN-2026-10-13.md` (sections 3-6 + audit).
Baseline: `main` @ 8955f73, 11/11 tests pass (`.venv/bin/python -m pytest -q`).

## 0. Rules for every engineer
- Edit **only the files you own** (table below). `app.py`, `README.md`, `requirements.txt` are frozen in wave 1.
- Every existing public name keeps its signature and behaviour. New parameters are keyword-only with defaults.
- No new pip dependency (stdlib + what `requirements.txt` already has).
- Do not import a name that another wave-1 engineer is still adding at module import time. Use duck typing or a lazy import inside the function, so tests pass in any merge order.
- No network in unit tests (fake the stream / HTTP). Tests go in a new file per owner: `tests/test_<area>.py`. Do not edit `tests/test_chatbot.py`.
- Never print or write the API key. Only `OPENROUTER_API_KEY` from the environment.
- No commit, no push. Leave changes in the working tree.

| Owner | Files | Tests |
|---|---|---|
| Retrieval | `search.py` | `tests/test_search.py` |
| Tokens | `answer.py`, `llm.py`, new `cache.py`, new `budget.py` | `tests/test_answer.py`, `tests/test_llm.py`, `tests/test_cache_budget.py` |
| Crawler | `scraper.py`, `docs.py` | `tests/test_scraper.py`, `tests/test_docs.py` |
| Eval | new `eval/` | `eval/` only (runner smoke test in `tests/test_eval.py`) |
| Integrator (wave 2) | `app.py` | `tests/test_app.py` |

## 1. Current public API (frozen, keep working)

**scraper.py**
- `@dataclass Page(url: str, title: str, text: str)`, property `words -> int`
- `class BlockedURL(ValueError)`
- `check_public_url(url: str) -> None` (raises `BlockedURL`)
- `fetch(session, url, max_redirects=3) -> tuple[str, str] | None` (final_url, html)
- `crawl(start_url: str, max_pages: int = 15, on_page: Callable[[int, str], None] | None = None) -> Iterator[Page]`
- private but tested: `_scope(url) -> (host, path_prefix)`, `_links(html, base) -> list[str]`
- constants: `USER_AGENT, MAX_BYTES, TIMEOUT_S, DELAY_S, SKIP_EXT`

**docs.py**
- `read_pdfs(files) -> tuple[list[Page], list[str]]` (pages, friendly problem strings); `MAX_PAGES = 300`

**search.py**
- `@dataclass Passage(url: str, title: str, text: str)`; `chunk(page) -> list[Passage]`; `CHUNK_CHARS = 900`
- `Index(pages)`, attrs `passages, vocab, bm25`; `Index.top(question, k=6) -> list[Passage]` (passages sharing at least one token, BM25+ order)

**answer.py**
- `SYSTEM` (format with `site`), `class AnswerError(Exception)`, `ai_available() -> bool`
- `stream_answer(site, question, passages, history, stream=llm.stream_text) -> Iterator[str]` (calls `stream(messages)` with one argument)
- `extractive_answer(passages, n=3) -> str`

**llm.py**
- `class LLMError(Exception)`, `api_key()`, `available()`, `models()`, `reasoning(model)`, `DEFAULT_MODELS`, `REASONING`, `RETRYABLE`
- `complete_json(messages, schema, name="result", timeout=60) -> tuple[dict, str]`
- `stream_text(messages, timeout=90) -> Iterator[str]`

**extract.py** (no owner in wave 1, do not edit): `parse_fields(text)`, `extract_rows(url, text, fields, complete=llm.complete_json)`

## 2. search.py (retrieval engineer)

```python
TOP_K = 4
MIN_COVERAGE = 0.34      # tune with the eval; share of question content words found
REL_CUTOFF = 0.5         # keep hits scoring >= REL_CUTOFF * best score

@dataclass
class Hit:
    passage: Passage
    score: float          # BM25+ score on the searched query
    coverage: float       # 0..1, idf-weighted share of the question's content words (stopwords removed) in this passage

@dataclass
class SearchResult:
    question: str         # as asked
    search_query: str     # text actually searched (question, + prev_question, + translation)
    hits: list[Hit]       # passed both cutoffs, best first, len <= k
    lang: str             # detect_lang(question)
    translated: bool = False
    @property
    def matched(self) -> bool: ...        # bool(hits)
    @property
    def passages(self) -> list[Passage]: ...

def normalize_text(text: str) -> str      # NFKC + lower; Arabic: drop tashkeel/tatweel, unify أإآ->ا, ى->ي, ة->ه
def detect_lang(text: str) -> str         # "ar" if Arabic letters dominate, else "latin"; local, no model
def trim_passage(p: Passage, query: str, max_chars: int = 600, context: int = 1) -> Passage
    # keep sentences containing query content words + `context` sentences each side, joined with " … ",
    # cut to max_chars on a word boundary; same url/title; returns p unchanged if nothing matches

Translate = Callable[[str, str], str]     # (question, target_lang) -> translated question; never raises

class Index:
    lang: str                             # new attr: detect_lang over the passages (majority)
    def top(self, question: str, k: int = TOP_K) -> list[Passage]   # unchanged behaviour, default k 4
    def search(self, question: str, k: int = TOP_K, *, prev_question: str | None = None,
               translate: Translate | None = None) -> SearchResult
```
Behaviour of `search`:
- Tokens use `normalize_text` plus light Arabic prefix stripping (`ال`, and `و/ف/ب/ك/ل` + `ال`) for tokens of 4+ letters. Apply the same to passages and queries.
- `prev_question`: added to the query at lower weight than the current question (for "how much is it?"). Coverage is measured on the current question's content words, falling back to the previous question's when the current one has none.
- `translate`: called only when `detect_lang(question) != self.lang`; search runs on the translation (plus the original), `translated=True`. Without `translate`, a cross-language question may return `matched=False`.
- Off-topic question (e.g. "zzzz unrelated", "who won the 2022 world cup" on the shop fixture) -> `matched is False`.
- Pure function of the index: no network, no `llm` import.

## 3. answer.py + llm.py + cache.py + budget.py (tokens engineer)

**llm.py**
```python
@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0     # OpenRouter usage.cost when present, else price table estimate
    model: str = ""
    calls: int = 0            # model calls made (0 for no-match / cache / extractive)

def stream_text(messages, timeout=90, *, max_tokens: int = 4000, deadline_s: float | None = None,
                on_usage: Callable[[Usage], None] | None = None, session_id: str | None = None) -> Iterator[str]
def translate_query(text: str, target_lang: str, *, on_usage=None, timeout: int = 10) -> str
    # tiny call (Flash-Lite, max_tokens ~80); returns `text` unchanged on any failure
```
- Network errors inside `iter_lines` never escape: before any text -> next model; after text -> the existing "(The answer was interrupted...)" line.
- Add 403 to the fallback codes. Request usage in the stream and call `on_usage` once at the end.
- `deadline_s`: stop trying models after this many seconds and raise `LLMError`.

**cache.py**
```python
def source_id(pages: list[Page]) -> str                 # sha1 over sorted (url, text), 16 hex chars
def normalize_question(q: str) -> str                   # NFKC, lower, strip punctuation, collapse spaces
class AnswerCache:                                      # thread-safe, in memory, LRU
    def __init__(self, max_entries: int = 500, ttl_s: int = 86_400, clock=time.time): ...
    def get(self, source: str, question: str) -> "AnswerResult | None"
    def put(self, source: str, question: str, result: "AnswerResult") -> None
```

**budget.py** (one shared instance per process; UTC day; thread-safe)
```python
@dataclass
class BudgetDecision:
    allowed: bool
    reason: str               # "ok" | "session_cap" | "ip_cap" | "global_budget"
    message: str              # friendly visitor text, "" when ok; never says "reload"

class Budget:
    def __init__(self, session_cap: int = 12, ip_cap: int = 30, daily_usd: float = 1.0,
                 ip_crawl_cap: int = 5, clock=time.time): ...
    def check(self, session: str, ip: str) -> BudgetDecision
    def record(self, session: str, ip: str, usage: Usage) -> None     # counts 1 question when usage.calls > 0, adds cost
    def check_crawl(self, ip: str) -> BudgetDecision                  # reason "ip_crawl_cap"
    def record_crawl(self, ip: str) -> None
    def snapshot(self) -> dict                                       # {"day", "spent_usd", "questions", "crawls"}
```

**answer.py**
```python
ANSWER_MAX_TOKENS = 1500
HISTORY_TURNS = 2
HISTORY_ANSWER_CHARS = 300
NOT_FOUND_TEXT = "I couldn't find that on this site."   # fixed, 0 tokens

@dataclass
class AnswerResult:
    text: str                  # final text shown, FOLLOWUPS line removed
    status: str                # "answered" | "no_match" | "cached" | "extractive" | "error"
    passages: list[Passage]    # trimmed passages, numbered [1..n] as sent to the model
    cited: list[int]           # valid 1-based numbers found in text; accepts [1], [1, 2], [1][2]
    followups: list[str]       # 0-3 short questions
    usage: Usage               # zeros unless a model was called
    note: str = ""             # why extractive/error (budget message, "AI busy"), "" otherwise

def trim_history(history: list[dict], turns: int = HISTORY_TURNS, answer_chars: int = HISTORY_ANSWER_CHARS) -> list[dict]
    # last `turns` user+assistant pairs; assistant text cut to answer_chars, "**Sources**" block and [n] marks removed
def split_followups(text: str) -> tuple[str, list[str]]   # "...\nFOLLOWUPS: a | b | c" -> ("...", ["a","b","c"])
def parse_citations(text: str, n_passages: int) -> list[int]

class AnswerRun:
    def __iter__(self) -> Iterator[str]   # streamed text chunks; never yields any part of the FOLLOWUPS line
    result: AnswerResult | None           # set when iteration ends (also for no_match/cached/extractive, which yield the whole text once)

def ask(site: str, question: str, found, history: list[dict], *,
        source: str, session: str, ip: str,
        cache: "AnswerCache | None" = None, budget: "Budget | None" = None,
        stream=llm.stream_text, trim=None) -> AnswerRun
    # found: a search.SearchResult (duck typed: .matched, .passages, .search_query)
    # trim: (Passage, query) -> Passage, default lazy search.trim_passage
```
Order inside `ask` (one model call at most):
1. `not found.matched` -> `status="no_match"`, `text=NOT_FOUND_TEXT`, 0 calls, not cached, not counted.
2. Cache: only when `history` is empty (a follow-up depends on context). Hit -> `status="cached"`, usage zero.
3. `budget.check(session, ip)` denied -> `status="extractive"`, `text=extractive_answer(passages)`, `note=decision.message`.
4. Model: system prompt asks for about 120 words and a final `FOLLOWUPS: a | b | c` line; messages = system + `trim_history(history)` + excerpts of up to 4 trimmed passages + question. Calls `stream(messages, max_tokens=ANSWER_MAX_TOKENS, on_usage=...)`. On `LLMError` -> `status="error"`, extractive text, `note="AI busy"`; nothing from the fallback goes into history.
5. After success: `budget.record(...)`; `cache.put(...)` when history was empty.

`stream_answer` stays as it is (old callers and tests).

## 4. scraper.py + docs.py (crawler engineer)

```python
class CrawlError(Exception):
    code: str        # "bad_url" | "blocked" | "not_found" | "unreachable" | "timeout" | "robots" | "no_text" | "js_only"
    message: str     # friendly, visitor-facing, no stack, no internals
    def __init__(self, code: str, message: str): ...
class BlockedURL(CrawlError, ValueError)     # same name, still a ValueError; code "blocked" (or "bad_url" for scheme/port)

MAX_FETCHES_FACTOR = 3      # fetch attempts <= max_pages * 3
MAX_SECONDS = 60            # whole crawl
ROBOTS_MAX_BYTES = 500_000

def normalize_url(url: str) -> str     # strip, add "https://" when no scheme, reject spaces/empty host/non-numeric port
                                       # with CrawlError("bad_url", ...); never ValueError
@dataclass
class CrawlReport:
    start_url: str            # normalized
    pages: list[Page]
    fetched: int              # HTTP fetch attempts
    stopped: str              # "done" | "max_pages" | "max_fetches" | "time_limit"
    js_only: bool             # HTML came back but almost no text and JS-app markers
    warnings: list[str]       # friendly notes for the visitor

def read_site(url: str, max_pages: int = 15, on_page=None, *, max_seconds: float = MAX_SECONDS) -> CrawlReport
    # raises only CrawlError; a site with 0 pages returns a report (pages=[]) with js_only/warnings set
def looks_js_only(html: str, text: str) -> bool
```
- `crawl()` keeps its signature, now uses the same caps and raises only `CrawlError`.
- Fix: unknown charset (`LookupError`) -> decode as utf-8 with replace; robots.txt size cap; thin pages count toward the fetch cap.
- `docs.read_pdfs` keeps its return shape. Wrap each page's text extraction; a PDF with pages but no text and with images -> problem `"{name}: looks scanned (images only). Client versions add OCR."`; also expose `is_scanned_pdf(reader) -> bool`.
- Out of scope in wave 1 (note only): DNS-rebinding gap between check and fetch.

## 5. eval/ (eval engineer)

```
eval/fixtures/<name>.json     # frozen pages: [{"url","title","text"}]; no live crawl during eval
eval/questions.jsonl          # 26 rows
eval/run.py                   # runner
eval/results/<tag>.json       # summary (committed later); *.raw = raw model output (gitignored)
eval/BASELINE.md              # baseline numbers, short
```
- Fixtures: `python_tutorial` (frozen 12-page crawl of docs.python.org/3/tutorial/), `shop` (small English shop: shipping, returns, payment, contact), `shop_ar` (2-3 short Arabic pages, handwritten).
- Question row: `{"id","fixture","question","prev": null|{"question","answer"},"lang","answerable": bool,"expect_urls": [...],"expect_any": [phrases]}`.
- Mix: 18 single questions (at least 3 unanswerable), 5 follow-ups (with `prev`), 3 in French or Arabic (at least 1 Arabic question on an English fixture, 1 on `shop_ar`).
- `run.py --mode retrieval|full --tag NAME [--repo PATH]`:
  - retrieval (no key): hit@4 (an expected url in passages), no-match accuracy (unanswerable -> nothing found / `matched False`), passages count.
  - full (key from env): answer correct (`expect_any` phrase, or NOT_FOUND for unanswerable), model calls, input/output tokens (real usage when the API returns it, else chars/4), cost, latency.
  - Adapter: uses `Index.search` + `answer.ask` when present, else `Index.top(q, 6)` + `stream_answer` (the old app flow, with the old full history). So the same runner scores `main` and `upgrade-v2`.
- Baseline: `git worktree add ../site-chatbot-baseline main` (read only, no edits there), run both modes, write numbers to `eval/BASELINE.md`.

## 6. How the integrator wires it (wave 2, for reference)

```python
report = scraper.read_site(url, max_pages)            # except CrawlError as e: show e.message
index = search.Index(report.pages); source = cache.source_id(report.pages)
found = index.search(q, prev_question=last_q, translate=lambda t, lang: llm.translate_query(t, lang, on_usage=...))
run = answer.ask(site, q, found, history, source=source, session=sid, ip=ip, cache=CACHE, budget=BUDGET)
st.write_stream(iter(run)); res = run.result         # res.text, res.cited, res.followups, res.usage, res.status
```

## 7. Wave 1 done when
- Old 11 tests + new tests pass; no file outside your row changed (`git status`).
- Eval: retrieval hit@4 >= baseline; unanswerable -> 0 model calls; turn-4 input under ~1.2k tokens; Arabic question on the English fixture finds the right passage when `translate` is given.
