"""Structured extraction: turn page text into table rows with the fields the user names."""
from __future__ import annotations

import re

import llm

MAX_FIELDS = 8
MAX_CHARS = 12_000  # per page sent to the model


def parse_fields(text: str) -> list[tuple[str, str]]:
    """'Product name, price, in stock?' -> [('product_name', 'Product name'), ...]"""
    out, seen = [], set()
    for raw in text.split(","):
        label = raw.strip()
        key = re.sub(r"\W+", "_", label.lower()).strip("_")
        if label and key and key not in seen:
            out.append((key, label))
            seen.add(key)
    return out[:MAX_FIELDS]


def _schema(fields: list[tuple[str, str]]) -> dict:
    row = {"type": "object", "properties": {k: {"type": "string"} for k, _ in fields}, "required": [k for k, _ in fields], "additionalProperties": False}
    return {"type": "object", "properties": {"rows": {"type": "array", "items": row}}, "required": ["rows"], "additionalProperties": False}


PROMPT = """Extract every item on this web page that has these fields: {fields}.
Return one row per item, in page order. Copy values exactly as written on the page.
Use an empty string when a field is missing for an item. Never invent values.
If the page has no such items, return an empty list.

Page: {url}
---
{text}"""


def extract_rows(url: str, text: str, fields: list[tuple[str, str]], complete=llm.complete_json) -> list[dict]:
    labels = ", ".join(f'"{k}" ({label})' for k, label in fields)
    data, _model = complete([{"role": "user", "content": PROMPT.format(fields=labels, url=url, text=text[:MAX_CHARS])}], _schema(fields), "rows")
    rows = []
    for r in data.get("rows", []):
        if isinstance(r, dict):
            clean = {label: str(r.get(k, "") or "").strip() for k, label in fields}
            if any(clean.values()):
                rows.append(clean)
    return rows
