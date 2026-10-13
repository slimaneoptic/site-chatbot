"""Split pages into passages and find the ones that best match a question (BM25).

`Index.top` is the old keyword search. `Index.search` adds what a chat needs: Arabic normalization,
the previous question for follow-ups, a relevance cutoff (so an off-topic question finds nothing and
costs no model call) and an optional query translation when the question and the site use different
scripts or languages. Pure functions of the index: no network, no `llm` import.
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
SPLIT_BELOW_COVERAGE = 0.6  # a multi-part question whose best hit covers less is also searched part by part
PART_MIN_COVERAGE = 0.25   # clause cutoff once the whole question found the site (it is on topic)

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
_SENT = re.compile(r"(?<=[.!?؟。])\s+|\n\s*\n")  # sentence ends and blank lines; single newlines are often hard wraps
_SENT_MAX = 400  # a "sentence" longer than this (a list without punctuation) is split on its lines instead


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


# Common Arabic unit words whose plural or case ending is not a prefix (broken plurals, tanween alif):
# "كم يوما" must find "4 أيام", "بكم درهم" must find "30 درهما". Keys are normalized and prefix-stripped.
_AR_FORMS = {
    "ايام": "يوم", "يوما": "يوم", "اياما": "يوم", "ساعات": "ساعه", "اسابيع": "اسبوع", "اسبوعا": "اسبوع",
    "شهور": "شهر", "اشهر": "شهر", "شهرا": "شهر", "دراهم": "درهم", "درهما": "درهم", "دقايق": "دقيقه",
    "دقائق": "دقيقه", "سنوات": "سنه", "طلبات": "طلب", "منتجات": "منتج",
}


def _stem(t: str) -> str:
    """Light, symmetric stemming: Arabic article prefixes on 4+ letter tokens and a few unit plurals, Latin
    plural -s/-ies. A bare conjunction و is not stripped ("الواتساب" -> "واتساب" must still equal "واتساب");
    و + a stopword ("وكم") is a stopword, see _is_stop."""
    n = len(t)
    if n < 4:
        return t
    if not t.isascii():
        for pre in _AR_PREFIXES:
            if t.startswith(pre) and n - len(pre) >= 2:
                t = t[len(pre):]
                break
        return _AR_FORMS.get(t, t)
    if t.isascii():
        if t.endswith("ies") and n > 4:
            return t[:-3] + "y"
        if t.endswith("s") and not t.endswith(("ss", "us", "is")):
            t, n = t[:-1], n - 1
        if t.endswith("ing") and n >= 7:  # "Can I track it?" finds "tracking"; "string", "thing", "during" stay
            t = t[:-3]
            if t[-1] == t[-2] and t[-1] not in "lsz":  # shipping -> ship, setting -> set; calling, missing keep
                t = t[:-1]
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
explain describe detail details detailed please everything long often soon take takes took taking
"""
_STOPWORDS = {_stem(t) for t in _WORD.findall(normalize_text(_STOP_RAW))}


def _is_stop(t: str) -> bool:
    """Stopword, or و/ف + an Arabic stopword ("وكم", "فهل": the 3-letter forms are not prefix-stripped)."""
    return t in _STOPWORDS or (len(t) > 1 and t[0] in "وف" and not t.isascii() and t[1:] in _STOPWORDS)


def _content(tokens: list[str]) -> list[str]:
    """Tokens without stopwords, first occurrence order, no duplicates."""
    seen: dict[str, None] = {}
    for t in tokens:
        if not _is_stop(t):
            seen.setdefault(t)
    return list(seen)


# Function words that mark a Latin-script language. Small closed sets, so a short question is enough
# ("Quels moyens de paiement acceptez-vous ?" -> fr). Words shared by two languages are left out.
_LATIN_MARKS = {
    "en": set("the and is are you your with for of to we our this that what how do does can which when where "
              "who why i my it there have has will would should".split()),
    "fr": set("le la les des du et est vous votre vos nous avec pour dans une sur quel quels quelle quelles "
              "comment combien pourquoi quand qui je peux puis-je sont ai avez acceptez".split()),
    "es": set("el los las del y es usted ustedes con para una por qué cómo cuánto cuándo dónde puedo hay "
              "tienen aceptan son".split()),
    "de": set("der die das und ist sie ich wir mit für ein eine wie was wann wo warum kann können gibt "
              "nicht auf ihr ihre".split()),
    "it": set("il gli della di che è come quanto quando dove posso sono avete con per una".split()),
    "pt": set("o os as da do dos das e é você vocês com para uma como quanto quando onde posso são aceitam "
              "não".split()),
}


def _latin_lang(text: str, limit: int = 400) -> str:
    """Best guess of a Latin-script text's language from function words; "" when nothing marks it."""
    words = re.findall(r"[a-zà-öø-ÿ]+(?:-[a-zà-öø-ÿ]+)?", (text or "").lower())[:limit]
    words += [h for w in words if "-" in w for h in w.split("-")]  # "acceptez-vous" -> acceptez, vous
    if not words:
        return ""
    score = {lang: sum(w in marks for w in words) for lang, marks in _LATIN_MARKS.items()}
    best = max(score, key=score.get)
    if score[best] == 0 or sorted(score.values())[-2] == score[best]:
        return ""  # unmarked or a tie: not sure, so no translation
    return best


def question_lang(text: str) -> str:
    """"ar", or a Latin language code ("en", "fr", "es", ...), or "" when a Latin question has no clear marks."""
    return "ar" if detect_lang(text) == "ar" else _latin_lang(text, 60)


def _site_lang(passages: list[Passage]) -> tuple[str, str]:
    """(script, language): ("ar", "ar") or ("latin", "en" | "fr" | "es" | ...). Majority over passages; empty -> latin/en."""
    if not passages:
        return "latin", "en"
    ar = sum(detect_lang(p.text) == "ar" for p in passages)
    if ar * 2 > len(passages):
        return "ar", "ar"
    votes: dict[str, int] = {}
    for p in passages[:50]:
        lang = _latin_lang(p.text)
        if lang:
            votes[lang] = votes.get(lang, 0) + 1
    return "latin", (max(votes, key=votes.get) if votes else "en")


# Clause breaks of a multi-part question: ? ; , and "and"/"et"/"y"/"und", Arabic "،" and و glued to a
# question word ("كم سعر التوصيل وكم يوما يستغرق؟" -> two parts).
_PART_SPLIT = re.compile(
    r"[?؟;!،,]+|\s+(?:and|et|y|und|e|also)\s+|"
    r"\s+(?=و(?:كم|ما|ماذا|هل|كيف|متى|متي|اين|أين|لماذا|من|بكم)(?:\s|$))",
    re.IGNORECASE)


def split_parts(question: str) -> list[str]:
    """The question's clauses that have a content word of their own; [] when it has fewer than two."""
    parts = [x.strip() for x in _PART_SPLIT.split(question or "") if x and x.strip()]
    parts = [x for x in parts if _content(_tokens(x))]
    return parts if len(parts) >= 2 else []


# ---------- trimming ----------

def _cut(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max(1, max_chars - 1)]
    sp = cut.rfind(" ")
    if sp > max_chars // 2:
        cut = cut[:sp]
    return cut.rstrip(" …") + "…"


def _sentences(text: str) -> list[str]:
    """Sentences with hard-wrapped lines joined ("...exit with a zero exit\nstatus." is one sentence)."""
    out = []
    for piece in _SENT.split(text or ""):
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) > _SENT_MAX and "\n" in piece:
            out += [ln.strip() for ln in piece.split("\n") if ln.strip()]
        else:
            out.append(re.sub(r"\s*\n\s*", " ", piece))
    return out


def trim_passage(p: Passage, query: str, max_chars: int = 600, context: int = 1) -> Passage:
    """Keep the sentences holding the query's content words plus `context` sentences each side, joined with
    " … ", cut to max_chars on a word boundary. Same url/title. Returns p unchanged when nothing matches."""
    q = _tokens(query)
    words = set(_content(q)) or set(q)
    if not words:
        return p
    sents = _sentences(p.text)
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
        cross = translate is not None and self.bm25 is not None and any(c.isalpha() for c in question) and self._foreign(question)
        well_covered = bool(res.hits) and max(h.coverage for h in res.hits) >= TRANSLATE_SKIP_COVERAGE
        if not cross or well_covered:  # one model call, and only when the question's own words do not find it
            return self._with_parts(res, question, k)
        try:
            tr = (translate(question, self.site_lang) or "").strip()
        except Exception:  # noqa: BLE001 - translate should never raise; a failure just means no translation
            tr = ""
        if not tr or _tokens(tr) == _tokens(question):
            return self._with_parts(res, question, k)
        res = self._search(question, k, prev_question, tr)
        res.lang, res.translated = lang, True
        return self._with_parts(res, tr, k, translated=True)

    def _with_parts(self, res: SearchResult, text: str, k: int, translated: bool = False) -> SearchResult:
        """Query expansion for multi-part and long questions ("the delivery price and how many days it takes"):
        when the whole question finds nothing or covers it poorly, search each clause on its own and add the
        best passages of every clause after the whole-question hits (no model call)."""
        if res.hits and max(h.coverage for h in res.hits) >= SPLIT_BELOW_COVERAGE:
            return res
        parts = split_parts(text)
        if not parts:
            return res
        cut = PART_MIN_COVERAGE if res.hits else MIN_COVERAGE  # an off-site question gets no looser bar
        found = [(self._search("", k, None, x, cut) if translated else self._search(x, k, None, "", cut)).hits
                 for x in parts]
        seen = {normalize_text(h.passage.text) for h in res.hits}
        for rank in range(k):
            for hits in found:
                if len(res.hits) >= k:
                    return res
                if rank < len(hits) and normalize_text(hits[rank].passage.text) not in seen:
                    seen.add(normalize_text(hits[rank].passage.text))
                    res.hits.append(hits[rank])
        return res

    def _foreign(self, question: str) -> bool:
        """Question in another language than the site: another script, or another Latin language
        (a French question on an English site). An unmarked Latin question counts as the site's language."""
        if detect_lang(question) != self.lang:
            return True
        q = question_lang(question)
        return bool(q) and self.lang == "latin" and q != self.site_lang

    def _gain(self, q: list[str]) -> list[float]:
        if not q:
            return [0.0] * len(self.passages)
        floor = sum(self.bm25.idf.get(t) or 0.0 for t in q) * self.bm25.delta  # what a passage with none of q gets
        return [float(s) - floor for s in self.bm25.get_scores(q)]

    def _search(self, question: str, k: int, prev: str | None, tr: str, min_cov: float | None = None) -> SearchResult:
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
        cand = [(i, c) for i, c in cand if c >= (MIN_COVERAGE if min_cov is None else min_cov)]
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
