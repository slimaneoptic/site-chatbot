"""Split pages into passages and find the ones that best match a question (BM25)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Plus

from scraper import Page

CHUNK_CHARS = 900


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


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class Index:
    """BM25+ keeps term weights positive, so tiny sites (a handful of passages) still rank well."""

    def __init__(self, pages: list[Page]):
        self.passages = [c for p in pages for c in chunk(p)]
        tokens = [_tokens(f"{c.title} {c.text}") for c in self.passages]
        self.vocab = [set(t) for t in tokens]
        self.bm25 = BM25Plus(tokens) if tokens else None

    def top(self, question: str, k: int = 6) -> list[Passage]:
        q = _tokens(question)
        if not self.bm25 or not q:
            return []
        scores = self.bm25.get_scores(q)
        hits = [i for i in range(len(scores)) if self.vocab[i] & set(q)]
        hits.sort(key=lambda i: scores[i], reverse=True)
        return [self.passages[i] for i in hits[:k]]
