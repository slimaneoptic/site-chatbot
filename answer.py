"""Answer questions from scraped passages only: an AI model via OpenRouter when a key is set,
extractive passages otherwise (or when every free model is busy)."""
from __future__ import annotations

import re
from typing import Iterator

import llm
from search import Passage

SYSTEM = """You are the website assistant for {site}. Answer the visitor's question using only the numbered excerpts from the site.

Cite the excerpts you used with their numbers in square brackets, like [2].
If the excerpts don't contain the answer, say you couldn't find it on the site and suggest where the visitor might look or who to contact. Don't guess and don't use outside knowledge.
Reply in the visitor's language, keep it short and practical, and use plain text or simple bullet lists."""


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
