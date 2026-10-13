"""Write the static examples file the front ends load at once (site/chat/examples.json).

It holds the example sources that GET /api/examples serves (eval/fixtures: label, url, kind, pages, page_list ...)
and, for each starter question, a REAL answer made by the engine through the real /api/chat code path. Each answer
keeps the exact shape of the live SSE stream, one key per event, so one renderer handles both:

    {"q": "...",
     "meta": {lang, dir, pages},
     "sources": [{n, title, url, passage, hl, hl_at, terms}],
     "notice": {code, message} | null,
     "text": "<all delta texts joined>",
     "followups": ["..."],
     "done": {ttfw_ms, cached, cost, left_today: null, status, cited, model, hl}}

`sources[].hl` is already the answer-based sentence (from done.hl), so a page can show it as is.

`ttfw_ms` and `cost` are the numbers measured on this run (the page shows these answers as "Cached · $0").
`left_today` is null: the export's own counter means nothing to a visitor (the live stream sends a number).
`kb_id` (ex-<fixture>) is the same id the live API pins, so a visitor can keep chatting about an example live.

Re-run it after changing the fixtures, eval/questions.jsonl, the prompts or the model (from the engine repo root;
16 answers cost about $0.01-0.03 with the default model):

    export OPENROUTER_API_KEY=<your key>
    .venv/bin/python scripts/export_examples.py                  # writes ../../../site/chat/examples.json
    .venv/bin/python scripts/export_examples.py --out /tmp/x.json --max-usd 0.05

The file is written only when every starter got a real AI answer ("answered"); --allow-partial writes it anyway.
It must stay under ~300 KB (the script refuses to write a bigger one).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import llm  # noqa: E402
from api import main  # noqa: E402
from budget import Budget  # noqa: E402

DEFAULT_OUT = ROOT.parents[2] / "site" / "chat" / "examples.json"
MAX_BYTES = 300_000
SUMMARY_KEYS = ("kb_id", "label", "kind", "url", "lang", "pages", "words", "page_list")


def parse_sse(text: str) -> list[tuple[str, object]]:
    out = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields:
            out.append((fields["event"], json.loads(fields.get("data", "null"))))
    return out


def to_answer(question: str, events: list[tuple[str, object]]) -> dict:
    """Fold one /api/chat stream into {q, meta, sources, notice, text, followups, done}."""
    ans = {"q": question, "meta": None, "sources": [], "notice": None, "text": "", "followups": [], "done": None}
    for name, data in events:
        if name == "delta":
            ans["text"] += data["text"]
        elif name in ("meta", "sources", "notice", "followups", "done"):
            ans[name] = data
        elif name == "error":
            raise RuntimeError(f"{question!r}: {data}")
    if ans["done"] is None:
        raise RuntimeError(f"{question!r}: the stream ended without done")
    for src in ans["sources"] or []:  # the answer-based highlight (done.hl) replaces the question-based guess
        better = (ans["done"].get("hl") or {}).get(str(src.get("n")))
        if better and better.get("hl"):
            src["hl"], src["hl_at"] = better["hl"], better.get("hl_at")
    return ans


def ask(client: TestClient, kb_id: str, question: str, session: str) -> dict:
    r = client.post("/api/chat", json={"kb_id": kb_id, "session_id": session,
                                       "messages": [{"role": "user", "content": question}]})
    return to_answer(question, parse_sse(r.text))


def build(client: TestClient, retries: int = 1, log=print) -> tuple[dict, list[str]]:
    problems: list[str] = []
    examples = []
    for ex in client.get("/api/examples").json()["examples"]:
        item = {k: ex.get(k) for k in SUMMARY_KEYS}
        item["starters"] = []
        for i, s in enumerate(ex["starters"]):
            for attempt in range(retries + 1):
                ans = ask(client, ex["kb_id"], s["q"], f"export-{ex['kb_id']}-{i}-{attempt}")
                if ans["done"]["status"] == "answered":
                    break
            d = ans["done"]
            d["left_today"] = None
            log(f"  {ex['kb_id']:<22} {d['status']:<10} {d['ttfw_ms']:>6} ms  ${d['cost']:.5f}  {s['q'][:60]}")
            if d["status"] != "answered":
                problems.append(f"{ex['kb_id']}: {s['q']!r} -> {d['status']}")
            item["starters"].append(ans)
        examples.append(item)
    models = sorted({a["done"].get("model") for e in examples for a in e["starters"] if a["done"].get("model")})
    data = {"version": 1, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "models": models, "examples": examples}
    return data, problems


def main_cli(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-usd", type=float, default=0.15, help="stop asking the model after this spend (default 0.15)")
    ap.add_argument("--allow-partial", action="store_true", help="write even if some starters got no AI answer")
    args = ap.parse_args(argv)
    if not llm.available():
        print("OPENROUTER_API_KEY is not set: the answers must be real. See the docstring.", file=sys.stderr)
        return 2
    app = main.create_app(budget=Budget(session_cap=100, ip_cap=100, daily_usd=args.max_usd, ip_crawl_cap=0,
                                        ip_upload_cap=0))
    t0 = time.monotonic()
    data, problems = build(TestClient(app))
    spent = app.state.budget.snapshot()["spent_usd"]
    body = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    n = sum(len(e["starters"]) for e in data["examples"])
    print(f"{n} answers, ${spent:.4f}, {time.monotonic() - t0:.0f} s, {len(body.encode()) / 1000:.0f} KB")
    if problems and not args.allow_partial:
        print("Not written, some starters have no AI answer:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    if len(body.encode()) > MAX_BYTES:
        print(f"Not written: over {MAX_BYTES // 1000} KB.", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(body + "\n", encoding="utf-8")
    tmp.replace(args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
