"""Answer questions from scraped passages only: an AI model via OpenRouter when a key is set,
extractive passages otherwise (or when every free model is busy)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Callable, Iterator

import llm
from search import Passage

SYSTEM = """You are the website assistant for {site}. Answer the visitor's question using only the numbered excerpts from the site.

Cite the excerpts you used with their numbers in square brackets, like [2].
If the excerpts don't contain the answer, say you couldn't find it on the site and suggest where the visitor might look or who to contact. Don't guess and don't use outside knowledge.
Reply in the visitor's language, keep it short and practical, and use plain text or simple bullet lists."""


ANSWER_MAX_TOKENS = 1500
HISTORY_TURNS = 2
HISTORY_ANSWER_CHARS = 300
MAX_PASSAGES = 4
NOT_FOUND_TEXT = "I couldn't find that on this site."  # fixed, 0 tokens
INTERRUPTED_NOTE = "(The answer was interrupted. Please ask again.)"

ANSWER_SYSTEM = """You are the website assistant for {site}. Answer the visitor's question using only the numbered excerpts from the site.

Cite the excerpts you used with their numbers in square brackets, like [2].
If the excerpts don't contain the answer, say you couldn't find it on the site and suggest where the visitor might look. Don't guess and don't use outside knowledge.
Reply in the visitor's language. Keep it short and practical: usually two to four sentences (under 80 words), or a short bullet list for steps.
Then end with one last line in this exact form, with up to 3 short questions (a few words each) the visitor could ask next that the excerpts answer, in the visitor's language:
FOLLOWUPS: question 1 | question 2 | question 3"""


class AnswerError(Exception):
    pass


def ai_available() -> bool:
    return llm.available()


def _excerpts(passages: list[Passage]) -> str:
    return "\n\n".join(f"[{i}] {p.title} ({p.url})\n{p.text}" for i, p in enumerate(passages, start=1))


def stream_answer(site: str, question: str, passages: list[Passage], history: list[dict], stream=llm.stream_text) -> Iterator[str]:
    """Yield answer text as it streams. `history` holds earlier {"role", "content"} turns."""
    messages = (
        [{"role": "system", "content": SYSTEM.format(site=site)}]
        + history[-6:]
        + [{"role": "user", "content": f"Excerpts from the site:\n\n{_excerpts(passages)}\n\nQuestion: {question}"}]
    )
    try:
        yield from stream(messages)
    except llm.LLMError as err:
        raise AnswerError(str(err)) from err


def extractive_answer(passages: list[Passage], n: int = 3) -> str:
    if not passages:
        return "I couldn't find anything about that on the scraped pages."
    parts = ["Here are the most relevant passages from the site:"]
    for i, p in enumerate(passages[:n], start=1):
        snippet = p.text[:420].rsplit(" ", 1)[0] + ("…" if len(p.text) > 420 else "")
        plain = re.sub(r"([\\`*_{}\[\]<>#+\-!|])", r"\\\1", snippet.replace("\n", " "))  # show text literally
        ref = f"[{p.title}]({p.url})" if p.url.startswith("http") else p.title
        parts.append(f"**[{i}] {ref}**\n\n> {plain}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- token-lean answers (wave 1)

@dataclass
class AnswerResult:
    text: str  # final text shown, FOLLOWUPS line removed
    status: str  # "answered" | "no_match" | "cached" | "extractive" | "error"
    passages: list[Passage]  # trimmed passages, numbered [1..n] as sent to the model
    cited: list[int]  # valid 1-based numbers found in text
    followups: list[str]  # 0-3 short questions
    usage: llm.Usage = field(default_factory=llm.Usage)  # zeros unless a model was called
    note: str = ""  # why extractive/error (budget message, "AI busy"), "" otherwise


_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_MARK_STRIP_RE = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")
# Marker of the follow-ups line: "FOLLOWUPS:" anywhere, or "Followups:" / "**Follow-ups:**" at a line start.
_FOLLOW_ANY_RE = re.compile(r"[*_]{0,2}FOLLOWUPS[*_]{0,2}[ \t]*:")
_FOLLOW_LINE_RE = re.compile(r"(?im)^[ \t>*_-]*follow-?ups[*_]{0,2}[ \t]*:")
_MARKER_WORD = "followups:"


def _find_marker(text: str) -> re.Match | None:
    found = [m for m in (_FOLLOW_ANY_RE.search(text), _FOLLOW_LINE_RE.search(text)) if m]
    return min(found, key=lambda m: m.start()) if found else None


def _holdback(text: str) -> int:
    """How many trailing chars could still become a FOLLOWUPS marker, so the stream holds them back."""
    for size in range(min(len(text), 16), 0, -1):
        tail = text[-size:]
        core = tail.lstrip(" \t>*_-").lower()
        if not core:
            return size  # only spaces or "**" so far
        word = core.rstrip("*_ \t").replace("-", "")
        before = text[-size - 1] if size < len(text) else "\n"
        if core[0] == "f" and _MARKER_WORD.startswith(word) and not before.isalnum():  # "of", "staff" are not held
            return size
    return 0


def split_followups(text: str) -> tuple[str, list[str]]:
    """"...\nFOLLOWUPS: a | b | c" -> ("...", ["a", "b", "c"])."""
    m = _find_marker(text)
    if not m:
        return text.rstrip(), []
    line = text[m.end():].split("\n", 1)[0]
    items = []
    for raw in line.split("|"):
        q = _MARK_STRIP_RE.sub("", raw).strip().strip("*_\"'“”«» ").lstrip("-•0123456789.) ").strip()
        if q and len(q) <= 150 and q not in items:
            items.append(q)
    return text[:m.start()].rstrip(), items[:3]


def parse_citations(text: str, n_passages: int) -> list[int]:
    """Valid 1-based excerpt numbers in order of first use; accepts [1], [1, 2] and [1][2]."""
    out: list[int] = []
    for m in _CITE_RE.finditer(text):
        for num in m.group(1).split(","):
            i = int(num)
            if 1 <= i <= n_passages and i not in out:
                out.append(i)
    return out


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip() + "…"


def _clean_answer(text: str) -> str:
    text = re.split(r"\*\*Sources\*\*", text, maxsplit=1)[0]
    text = split_followups(text)[0]
    return re.sub(r"[ \t]+", " ", _MARK_STRIP_RE.sub("", text)).strip()


def trim_history(history: list[dict], turns: int = HISTORY_TURNS, answer_chars: int = HISTORY_ANSWER_CHARS) -> list[dict]:
    """Last `turns` user+assistant pairs; assistant text cut to `answer_chars` with the
    "**Sources**" block and [n] marks removed. Returns new {"role", "content"} dicts."""
    msgs = [m for m in history or [] if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
    users = [i for i, m in enumerate(msgs) if m["role"] == "user"]
    if turns <= 0 or not users:
        return []
    out = []
    for m in msgs[users[-turns] if len(users) >= turns else users[0]:]:
        if m["role"] == "assistant":
            content = _cut(_clean_answer(m["content"]), answer_chars)
        else:
            content = _cut(m["content"].strip(), answer_chars)
        if content:
            out.append({"role": m["role"], "content": content})
    return out


def _default_trim(p: Passage, query: str) -> Passage:
    try:  # lazy: search.trim_passage arrives in the same wave
        from search import trim_passage
    except ImportError:
        return Passage(p.url, p.title, _cut(p.text, 600))
    return trim_passage(p, query)


def _build_messages(site: str, question: str, passages: list[Passage], history: list[dict]) -> list[dict]:
    return (
        [{"role": "system", "content": ANSWER_SYSTEM.format(site=site)}]
        + trim_history(history)
        + [{"role": "user", "content": f"Excerpts from the site:\n\n{_excerpts(passages)}\n\nQuestion: {question}"}]
    )


def _estimate(messages: list[dict], output: str) -> llm.Usage:
    prompt = sum(len(str(m.get("content", ""))) for m in messages)
    return llm.Usage(prompt_tokens=(prompt + 3) // 4, completion_tokens=(len(output) + 3) // 4, calls=1)


class AnswerRun:
    """Iterate to stream the answer text; `result` is set when iteration ends.
    no_match / cached / extractive / error answers yield their whole text once."""

    def __init__(self, site: str, question: str, found, history: list[dict], *, source: str, session: str, ip: str,
                 cache=None, budget=None, stream=llm.stream_text, trim: Callable[[Passage, str], Passage] | None = None):
        self._args = (site, question, found, list(history or []))
        self._source, self._session, self._ip = source, session, ip
        self._cache, self._budget, self._stream = cache, budget, stream
        self._trim = trim or _default_trim
        self._started = False
        self.result: AnswerResult | None = None
        self.messages: list[dict] = []  # what was sent to the model, for eval and debugging

    def __iter__(self) -> Iterator[str]:
        if self._started:  # a second loop replays the text, it never calls the model again
            return iter([self.result.text] if self.result else [])
        self._started = True
        return self._run()

    def _finish(self, result: AnswerResult) -> Iterator[str]:
        self.result = result
        yield result.text

    def _run(self) -> Iterator[str]:
        site, question, found, history = self._args
        if not getattr(found, "matched", False):
            yield from self._finish(AnswerResult(NOT_FOUND_TEXT, "no_match", [], [], [], llm.Usage()))
            return
        use_cache = self._cache is not None and not history
        if use_cache:
            hit = self._cache.get(self._source, question)
            if hit is not None:
                yield from self._finish(replace(hit, status="cached", usage=llm.Usage(), note=""))
                return
        query = getattr(found, "search_query", "") or question
        passages = [self._trim(p, query) for p in list(found.passages)[:MAX_PASSAGES]]
        if self._budget is not None:
            decision = self._budget.check(self._session, self._ip)
            if not decision.allowed:
                text = extractive_answer(passages)
                yield from self._finish(AnswerResult(text, "extractive", passages, parse_citations(text, len(passages)), [],
                                                     llm.Usage(), decision.message))
                return
        self.messages = messages = _build_messages(site, question, passages, history)
        yield from self._model(messages, passages, use_cache, question)

    def _model(self, messages: list[dict], passages: list[Passage], use_cache: bool, question: str) -> Iterator[str]:
        usages: list[llm.Usage] = []
        try:
            chunks = self._stream(messages, max_tokens=ANSWER_MAX_TOKENS, on_usage=usages.append)
        except TypeError:  # an older stream(messages) without the new keywords
            chunks = self._stream(messages)
        full, sent, stopped, failed = "", 0, False, False
        try:
            for chunk in chunks:
                full += chunk
                if stopped:
                    continue
                m = _find_marker(full)
                end = m.start() if m else len(full) - _holdback(full)
                if m:
                    stopped = True
                if end > sent:
                    yield full[sent:end]
                    sent = end
        except llm.LLMError:
            failed = True
        except GeneratorExit:  # the page stopped reading: still count what the model was asked
            getattr(chunks, "close", lambda: None)()
            self._done(split_followups(full), passages, usages[-1] if usages else _estimate(messages, full), False, False, question)
            raise
        if failed and not full:
            text = extractive_answer(passages)
            yield from self._finish(AnswerResult(text, "error", passages, parse_citations(text, len(passages)), [],
                                                 llm.Usage(), "AI busy"))
            return
        text, followups = split_followups(full)
        if failed:
            text += "\n\n" + INTERRUPTED_NOTE
        if not stopped and len(text) > sent:  # release what was held back
            yield text[sent:]
        elif stopped and failed:
            yield "\n\n" + INTERRUPTED_NOTE
        usage = usages[-1] if usages else _estimate(messages, full)
        self._done((text, followups), passages, usage, failed, use_cache, question)

    def _done(self, parts: tuple[str, list[str]], passages: list[Passage], usage: llm.Usage, failed: bool,
              use_cache: bool, question: str) -> None:
        text, followups = parts
        self.result = AnswerResult(text, "error" if failed else "answered", passages, parse_citations(text, len(passages)),
                                   followups, usage, "AI busy" if failed else "")
        if self._budget is not None:
            self._budget.record(self._session, self._ip, usage)
        if use_cache and not failed and text and INTERRUPTED_NOTE not in text:
            self._cache.put(self._source, question, self.result)


def ask(site: str, question: str, found, history: list[dict], *, source: str, session: str, ip: str,
        cache=None, budget=None, stream=llm.stream_text, trim=None) -> AnswerRun:
    """One answer, at most one model call. `found` is a search.SearchResult (duck typed:
    .matched, .passages, .search_query). Iterate the returned run, then read run.result."""
    return AnswerRun(site, question, found, history, source=source, session=session, ip=ip,
                     cache=cache, budget=budget, stream=stream, trim=trim)
