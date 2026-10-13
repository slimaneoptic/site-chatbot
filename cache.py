"""Shared in-memory answer cache: same site text + same question -> same answer, 0 tokens."""
from __future__ import annotations

import hashlib
import re
import threading
import time
import unicodedata
from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # no import at runtime: answer.py imports nothing from here either
    from answer import AnswerResult


def source_id(pages: list) -> str:
    """Stable id of a site's text: sha1 over the sorted (url, text) pairs, 16 hex chars."""
    h = hashlib.sha1()
    for url, text in sorted((p.url, p.text) for p in pages):
        h.update(url.encode("utf-8", "replace") + b"\0" + text.encode("utf-8", "replace") + b"\1")
    return h.hexdigest()[:16]


def normalize_question(q: str) -> str:
    """NFKC, lower case, punctuation removed, spaces collapsed."""
    q = unicodedata.normalize("NFKC", q or "").lower()
    q = "".join(" " if unicodedata.category(c).startswith(("P", "S")) else c for c in q)
    return re.sub(r"\s+", " ", q).strip()


class AnswerCache:
    """Thread-safe LRU with a time-to-live. Keys are (source id, normalized question)."""

    def __init__(self, max_entries: int = 500, ttl_s: int = 86_400, clock=time.time):
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self._clock = clock
        self._lock = threading.Lock()
        self._items: OrderedDict[tuple[str, str], tuple[float, object]] = OrderedDict()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def get(self, source: str, question: str) -> "AnswerResult | None":
        key = (source, normalize_question(question))
        if not key[1]:
            return None
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            stored_at, result = item
            if self._clock() - stored_at > self.ttl_s:
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return result

    def put(self, source: str, question: str, result: "AnswerResult") -> None:
        key = (source, normalize_question(question))
        if not key[1] or self.max_entries <= 0:
            return
        with self._lock:
            self._items[key] = (self._clock(), result)
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)
