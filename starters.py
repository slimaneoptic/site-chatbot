"""Starter questions for a newly read site (the first chips a visitor clicks).

One cheap model call (Flash-Lite, <= 250 output tokens, 3 s) proposes six questions from the page titles and first
sentences; each is kept only when the local search answers it well. Without the AI (no key, no budget, a failure)
or when fewer than 3 pass, questions built from page titles fill in, checked the same way.
"""
from __future__ import annotations

import re
from typing import Callable

import search

MAX_STARTERS = 4
MIN_STARTERS = 3
MIN_COVER = 0.5
DIGEST_CHARS = 6000  # about 1,500 tokens of titles + first sentences
CANDIDATES = 6

SCHEMA = {"type": "object", "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
          "required": ["questions"], "additionalProperties": False}

_LANG = {"ar": "Arabic", "en": "English", "fr": "French", "es": "Spanish", "de": "German", "it": "Italian",
         "pt": "Portuguese"}

TEMPLATES = {
    "en": ("What does {t} cover?", "How do I contact you?"),
    "fr": ("Que contient la page « {t} » ?", "Comment vous contacter ?"),
    "es": ("¿Qué incluye «{t}»?", "¿Cómo puedo contactarles?"),
    "ar": ("ماذا تتضمن صفحة {t}؟", "كيف يمكنني التواصل معكم؟"),
}

_CONTACT = re.compile(r"contact|contacter|nous-joindre|اتصل|تواصل|site-info", re.I)


def site_language(pages) -> str:
    """"ar", "en", "fr", ... from the pages' text (same rule as the search index)."""
    passages = [search.Passage(p.url, p.title, p.text[:2000]) for p in pages[:20]]
    script, lang = search._site_lang(passages)
    return "ar" if script == "ar" else lang


def digest(pages) -> str:
    """Page titles plus the first sentence of each page, about 1,500 tokens at most."""
    lines, size = [], 0
    for p in pages:
        first = next((s for s in search._sentences(p.text) if len(s.split()) >= 4), "")[:200]
        line = f"- {p.title}: {first}" if first else f"- {p.title}"
        if size + len(line) > DIGEST_CHARS:
            break
        lines.append(line)
        size += len(line)
    return "\n".join(lines)


def ask_model(label: str, lang: str, pages, quick: Callable) -> tuple[list[str], object]:
    """Up to 6 candidate questions from the model, and its Usage. `quick(messages, schema)` -> (data, usage)."""
    language = _LANG.get(lang, "English")
    messages = [
        {"role": "system", "content": (
            f"You write the suggested questions shown under a website's chat box. Write {CANDIDATES} short questions "
            f"(at most 8 words each) that a visitor of this site would really ask and that these pages answer. "
            f"Write them in {language}. Use the site's own words. No yes/no questions about the chat itself.")},
        {"role": "user", "content": f"Site: {label}\nPages:\n{digest(pages)}"},
    ]
    data, usage = quick(messages, SCHEMA)
    qs = data.get("questions") if isinstance(data, dict) else None
    out = []
    for q in qs or []:
        if isinstance(q, str):
            q = " ".join(q.split()).strip(" -•*\"'")
            if 3 <= len(q) <= 90 and q not in out:
                out.append(q)
    return out[:CANDIDATES], usage


def templates(pages, lang: str) -> list[str]:
    what, contact = TEMPLATES.get(lang, TEMPLATES["en"])
    out = []
    if any(_CONTACT.search(p.url) or _CONTACT.search(p.title) for p in pages):
        out.append(contact)
    for p in pages:
        t = (p.title or "").strip()
        if t and len(t.split()) <= 6 and len(t) <= 50 and not p.url.endswith("#site-info"):
            out.append(what.format(t=t))
    return list(dict.fromkeys(out))


def check(index, candidates: list[str], limit: int = MAX_STARTERS) -> list[str]:
    """Candidates the local search answers well (best hit covers >= half the question's words); questions whose
    best page differs come first, so the chips show different parts of the site."""
    ok = []
    for q in candidates:
        try:
            res = index.search(q)
        except Exception:  # noqa: BLE001
            continue
        if res.matched and max(h.coverage for h in res.hits) >= MIN_COVER:
            ok.append((q, res.hits[0].passage.url))
    first, rest, pages = [], [], set()
    for q, url in ok:
        (rest if url in pages else first).append(q)
        pages.add(url)
    return (first + rest)[:limit]


def pick(index, pages, label: str, ai_candidates: list[str] | None, lang: str | None = None) -> list[str]:
    """3-4 checked starters: the model's first, templates when fewer than 3 pass; [] when nothing passes."""
    lang = lang or site_language(pages)
    out = check(index, ai_candidates or [])
    if len(out) < MIN_STARTERS:
        for q in check(index, templates(pages, lang), limit=MAX_STARTERS * 3):
            if q not in out and len(out) < MAX_STARTERS:
                out.append(q)
    return out[:MAX_STARTERS]
