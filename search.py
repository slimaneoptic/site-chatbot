"""Split pages into passages and find the ones that best match a question (BM25).

`Index.top` is the old keyword search. `Index.search` adds what a chat needs: Arabic normalization,
the previous question for follow-ups, a relevance cutoff (so an off-topic question finds nothing and
costs no model call) and an optional query translation when the question and the site use different
scripts. Pure functions of the index: no network, no `llm` import.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

from rank_bm25 import BM25Plus

from scraper import Page

CHUNK_CHARS = 900
TOP_K = 4
MIN_COVERAGE = 0.34      # tune with the eval; share of question content words found
REL_CUTOFF = 0.5         # keep hits scoring >= REL_CUTOFF * best score
PREV_WEIGHT = 0.5        # weight of the previous question's words in a follow-up search
TRANSLATE_SKIP_COVERAGE = 0.6  # cross-language question already well covered as is -> no translation call

Translate = Callable[[str, str], str]  # (question, target_lang) -> translated question; never raises


@dataclass
class Passage:
    url: str
    title: str
    text: str


def chunk(page: Page) -> list[Passage]:
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", page.text) if p.strip()]
    out, buf = [], ""
    for p in paras:
        if buf and len(buf) + len(p) > CHUNK_CHARS:
            out.append(Passage(page.url, page.title, buf))
            tail = buf[-150:]
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail  # start the overlap on a whole word
            buf = tail + "\n" + p  # small overlap keeps context across the cut
        else:
            buf = f"{buf}\n{p}" if buf else p
    if buf:
        out.append(Passage(page.url, page.title, buf))
    return out


# ---------- text normalization ----------

_TASHKEEL = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭـ]")  # + tatweel
_AR_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ی": "ي", "ة": "ه",
    **{chr(0x0660 + d): str(d) for d in range(10)},  # Arabic-Indic digits -> 0-9 ("٣٠ يوم" matches "30 days")
    **{chr(0x06F0 + d): str(d) for d in range(10)},
})
_AR_PREFIXES = ("وال", "فال", "بال", "كال", "لل", "ال")  # longest first; "لل" is ل + ال
_WORD = re.compile(r"\w+")
_SENT = re.compile(r"(?<=[.!?؟。])\s+|\n+")


def normalize_text(text: str) -> str:
    """NFKC + lower; Arabic: drop tashkeel/tatweel, unify أإآ->ا, ى->ي, ة->ه (and Arabic-Indic digits -> 0-9)."""
    t = unicodedata.normalize("NFKC", text or "").lower()
    return _TASHKEEL.sub("", t).translate(_AR_MAP)


def _is_arabic(ch: str) -> bool:
    o = ord(ch)
    return 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF or 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF


def detect_lang(text: str) -> str:
    """"ar" if Arabic letters dominate, else "latin". Local, no model."""
    ar = other = 0
    for ch in text or "":
        if ch.isalpha():
            if _is_arabic(ch):
                ar += 1
            else:
                other += 1
    return "ar" if ar > other else "latin"


def _stem(t: str) -> str:
    """Light, symmetric stemming: Arabic article prefixes on 4+ letter tokens, Latin plural -s/-ies."""
    n = len(t)
    if n < 4:
        return t
    for pre in _AR_PREFIXES:
        if t.startswith(pre):
            return t[len(pre):] if n - len(pre) >= 2 else t
    if t.isascii():
        if t.endswith("ies") and n > 4:
            return t[:-3] + "y"
        if t.endswith("s") and not t.endswith(("ss", "us", "is")):
            return t[:-1]
    return t


def _tokens(text: str) -> list[str]:
    return [_stem(t) for t in _WORD.findall(normalize_text(text))]


_STOP_RAW = """
a an the and or but if of to in on at by for with from as is are was were be been being am do does did doing done
have has had having i me my mine we our ours us you your yours he him his she her it its they them their this that
these those what which who whom whose when where why how can could would should will shall may might must not no
yes there here than then so too very just also about into over under up out any some all each every more most much
many such only own same other s t don get got want need please tell know let like one thing things anything something
le la les un une des du de d l et ou est sont a ai as avez avons ce cet cette ces je tu il elle nous vous ils elles
mon ma mes ton ta tes son sa ses notre nos votre vos leur leurs que qui quoi quel quelle quels quelles quand comment
combien pourquoi ou dans sur pour par avec sans en au aux y ne pas plus se s qu c est-ce peut puis faire fait
ما ماذا هل كيف كم متي اين من في علي الي عن مع هذا هذه ذلك تلك هناك هو هي هم انا نحن انت انتم التي الذي الذين او
ثم لا لم لن ان قد كل اي بعد قبل عند حتي يمكن يمكنني ممكن اريد لماذا لكم لي لك عندكم
"""
_STOPWORDS = {_stem(t) for t in _WORD.findall(normalize_text(_STOP_RAW))}


def _content(tokens: list[str]) -> list[str]:
    """Tokens without stopwords, first occurrence order, no duplicates."""
    seen: dict[str, None] = {}
    for t in tokens:
        if t not in _STOPWORDS:
            seen.setdefault(t)
    return list(seen)


_EN_MARK = {"the", "and", "is", "are", "you", "your", "with", "for", "of", "to", "we", "our", "this", "that"}
_FR_MARK = {"le", "la", "les", "des", "du", "et", "est", "vous", "votre", "nous", "avec", "pour", "dans", "une", "sur"}


def _site_lang(passages: list[Passage]) -> tuple[str, str]:
    """(script, language): ("ar", "ar") or ("latin", "en" | "fr"). Majority over passages; empty -> latin/en."""
    if not passages:
        return "latin", "en"
    ar = sum(detect_lang(p.text) == "ar" for p in passages)
    if ar * 2 > len(passages):
        return "ar", "ar"
    en = fr = 0
    for p in passages[:50]:
        words = re.findall(r"[a-zà-ÿ]+", p.text.lower())[:400]
        en += sum(w in _EN_MARK for w in words)
        fr += sum(w in _FR_MARK for w in words)
    return "latin", ("fr" if fr > en else "en")


# ---------- trimming ----------

def _cut(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max(1, max_chars - 1)]
    sp = cut.rfind(" ")
    if sp > max_chars // 2:
        cut = cut[:sp]
    return cut.rstrip(" …") + "…"


def trim_passage(p: Passage, query: str, max_chars: int = 600, context: int = 1) -> Passage:
    """Keep the sentences holding the query's content words plus `context` sentences each side, joined with
    " … ", cut to max_chars on a word boundary. Same url/title. Returns p unchanged when nothing matches."""
    q = _tokens(query)
    words = set(_content(q)) or set(q)
    if not words:
        return p
    sents = [s.strip() for s in _SENT.split(p.text) if s and s.strip()]
    found = [i for i, s in enumerate(sents) if words & set(_tokens(s))]
    if not found:
        return p

    def build(ctx: int) -> str:
        keep = sorted({j for i in found for j in range(max(0, i - ctx), min(len(sents), i + ctx + 1))})
        groups: list[list[str]] = []
        last = None
        for j in keep:
            if last is not None and j == last + 1:
                groups[-1].append(sents[j])
            else:
                groups.append([sents[j]])
            last = j
        return " … ".join(" ".join(g) for g in groups)

    text = build(context)
    if len(text) > max_chars and context > 0:  # too long: matched sentences first, drop the context
        text = build(0)
    return Passage(p.url, p.title, _cut(text, max_chars))


# ---------- search ----------

@dataclass
class Hit:
    passage: Passage
    score: float     # BM25+ score on the searched query, minus the zero-match floor BM25+ gives every passage (same order)
    coverage: float  # 0..1, idf-weighted share of the question's content words (stopwords removed) in this passage


@dataclass
class SearchResult:
    question: str        # as asked
    search_query: str    # text actually searched (question, + prev_question, + translation)
    hits: list[Hit] = field(default_factory=list)  # passed both cutoffs, best first, len <= k
    lang: str = "latin"  # detect_lang(question)
    translated: bool = False

    @property
    def matched(self) -> bool:
        return bool(self.hits)

    @property
    def passages(self) -> list[Passage]:
        return [h.passage for h in self.hits]


class Index:
    """BM25+ keeps term weights positive, so tiny sites (a handful of passages) still rank well."""

    def __init__(self, pages: list[Page]):
        self.passages = [c for p in pages for c in chunk(p)]
        tokens = [_tokens(f"{c.title} {c.text}") for c in self.passages]
        self.vocab = [set(t) for t in tokens]
        self.bm25 = BM25Plus(tokens) if any(tokens) else None
        self.lang, self.site_lang = _site_lang(self.passages)  # "ar"|"latin", and "ar"|"en"|"fr" for translation

    def top(self, question: str, k: int = TOP_K) -> list[Passage]:
        q = _tokens(question)
        if not self.bm25 or not q:
            return []
        scores = self.bm25.get_scores(q)
        hits = [i for i in range(len(scores)) if self.vocab[i] & set(q)]
        hits.sort(key=lambda i: scores[i], reverse=True)
        return [self.passages[i] for i in hits[:k]]

    def search(self, question: str, k: int = TOP_K, *, prev_question: str | None = None,
               translate: Translate | None = None) -> SearchResult:
        question = question or ""
        lang = detect_lang(question)
        res = self._search(question, k, prev_question, "")
        res.lang = lang
        cross = translate is not None and self.bm25 is not None and lang != self.lang and any(c.isalpha() for c in question)
        well_covered = bool(res.hits) and max(h.coverage for h in res.hits) >= TRANSLATE_SKIP_COVERAGE
        if not cross or well_covered:  # one model call, and only when the question's own words do not find it
            return res
        try:
            tr = (translate(question, self.site_lang) or "").strip()
        except Exception:  # noqa: BLE001 - translate should never raise; a failure just means no translation
            tr = ""
        if not tr or _tokens(tr) == _tokens(question):
            return res
        res = self._search(question, k, prev_question, tr)
        res.lang, res.translated = lang, True
        return res

    def _gain(self, q: list[str]) -> list[float]:
        if not q:
            return [0.0] * len(self.passages)
        floor = sum(self.bm25.idf.get(t) or 0.0 for t in q) * self.bm25.delta  # what a passage with none of q gets
        return [float(s) - floor for s in self.bm25.get_scores(q)]

    def _search(self, question: str, k: int, prev: str | None, tr: str) -> SearchResult:
        query = " ".join(s.strip() for s in (question, prev or "", tr) if s and s.strip())
        out = SearchResult(question, query)
        if not self.bm25 or k <= 0:
            return out
        cur, prv, trt = _tokens(question), _tokens(prev or ""), _tokens(tr)
        cur_c, prv_c, tr_c = _content(cur), _content(prv), _content(trt)
        main = (cur_c or cur) + (tr_c or trt)
        cov_sets = [s for s in (cur_c or prv_c, tr_c) if s]  # current words; previous ones when it has none
        if not cov_sets or not (main or prv_c):
            return out

        g_main, g_prev = self._gain(main), self._gain(prv_c)
        gain = [a + PREV_WEIGHT * b for a, b in zip(g_main, g_prev)]
        qset = set(main) | set(prv_c)
        unknown = math.log(len(self.passages) + 1)  # a word the site never uses weighs like the rarest one

        def coverage(i: int) -> float:
            best = 0.0
            for words in cov_sets:
                w = {t: self.bm25.idf.get(t, unknown) for t in words}
                total = sum(w.values())
                if total > 0:
                    best = max(best, sum(v for t, v in w.items() if t in self.vocab[i]) / total)
            return best

        cand = [(i, coverage(i)) for i in range(len(self.passages)) if self.vocab[i] & qset and gain[i] > 1e-9]
        cand = [(i, c) for i, c in cand if c >= MIN_COVERAGE]
        if not cand:
            return out
        best = max(gain[i] for i, _ in cand)
        cand = sorted((x for x in cand if gain[x[0]] >= REL_CUTOFF * best), key=lambda x: gain[x[0]], reverse=True)
        seen: set[str] = set()
        for i, c in cand:
            key = normalize_text(self.passages[i].text)
            if key in seen:  # same text on two URLs (shared footer, ?ref= copies)
                continue
            seen.add(key)
            out.hits.append(Hit(self.passages[i], round(gain[i], 4), round(c, 4)))
            if len(out.hits) >= k:
                break
        return out
