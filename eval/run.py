"""Score the site chatbot on frozen pages: retrieval, answers, tokens, cost and latency.

    python eval/run.py --mode retrieval --tag NAME [--code-dir PATH]   # no API key, no network
    python eval/run.py --mode full --tag NAME [--code-dir PATH]        # needs OPENROUTER_API_KEY

--code-dir (alias --repo) points at the chatbot code to score (default: this repo), so the
same runner scores `main` (old flow: Index.top(q, 6) + stream_answer with full history) and
the upgrade (Index.search + answer.ask when they exist).
Writes eval/results/<tag>.json (metrics, no answer text) and <tag>.raw (answers, gitignored).
"""
from __future__ import annotations

import argparse
import importlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MODULES = ("scraper", "docs", "search", "llm", "answer", "extract", "cache", "budget")
NO_ANSWER_MARKERS = (
    "couldn't find", "could not find", "can't find", "cannot find", "didn't find", "did not find",
    "no information", "not covered", "doesn't cover", "does not cover", "isn't covered", "don't cover",
    "doesn't mention", "does not mention", "not mentioned", "not on the site", "no mention",
    "لم أجد", "لم أتمكن من العثور", "لم أعثر", "لا تتوفر", "لا يوجد", "n'ai pas trouvé", "ne trouve pas",
)


# ---- loading the code under test ------------------------------------------------------------

def load_code(code_dir: Path) -> dict:
    """Import the chatbot modules from code_dir (dropping copies loaded from anywhere else)."""
    code_dir = Path(code_dir).resolve()
    for name in MODULES:
        mod = sys.modules.get(name)
        if mod is not None and Path(getattr(mod, "__file__", "") or "").resolve().parent != code_dir:
            del sys.modules[name]
    if str(code_dir) in sys.path:
        sys.path.remove(str(code_dir))
    sys.path.insert(0, str(code_dir))
    mods = {name: importlib.import_module(name) for name in ("scraper", "search", "llm", "answer")}
    return mods


def git_rev(code_dir: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(code_dir), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "-C", str(code_dir), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return ""


# ---- usage spy: reads OpenRouter's usage chunk without changing the code under test ----------

class UsageSpy:
    """Stands in for the `requests` module inside llm.py. It asks OpenRouter to include usage in
    streamed answers and reads it from the stream; every other attribute is the real module."""

    def __init__(self, real):
        self._real = real
        self.reset()

    def reset(self):
        self.posts = 0
        self.ok_posts = 0
        self.usage: dict | None = None
        self.prompt_chars = 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    def post(self, url, *args, json=None, **kwargs):
        self.posts += 1
        if isinstance(json, dict):
            self.prompt_chars += sum(len(str(m.get("content", ""))) for m in json.get("messages", []))
            if json.get("stream"):
                json = {**json, "usage": {"include": True}}
        resp = self._real.post(url, *args, json=json, **kwargs)
        if getattr(resp, "status_code", 0) == 200:
            self.ok_posts += 1
        if isinstance(json, dict) and json.get("stream") and hasattr(resp, "iter_lines"):
            resp.iter_lines = self._sniff(resp.iter_lines)
        return resp

    def _sniff(self, iter_lines):
        spy = self

        def wrapped(*args, **kwargs):
            for line in iter_lines(*args, **kwargs):
                text = line.decode("utf-8", "replace") if isinstance(line, bytes) else (line or "")
                if text.startswith("data:") and '"usage"' in text:
                    try:
                        usage = __import__("json").loads(text[5:].strip(), strict=False).get("usage")
                    except ValueError:
                        usage = None
                    if usage:
                        spy.usage = usage
                yield line
        return wrapped


def _usage_fields(result_usage, spy: UsageSpy | None, est_in: int, out_text: str) -> dict:
    """Prefer usage reported by the code under test, then the spied API usage, else chars/4."""
    if result_usage is not None and getattr(result_usage, "calls", 0) and (result_usage.prompt_tokens or result_usage.completion_tokens):
        return {"input_tokens": result_usage.prompt_tokens, "output_tokens": result_usage.completion_tokens,
                "cost_usd": round(result_usage.cost_usd, 6), "model": result_usage.model, "usage_source": "code"}
    if spy is not None and spy.usage:
        u = spy.usage
        return {"input_tokens": int(u.get("prompt_tokens") or 0), "output_tokens": int(u.get("completion_tokens") or 0),
                "reasoning_tokens": int((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
                "cost_usd": round(float(u.get("cost") or 0.0), 6), "usage_source": "api"}
    return {"input_tokens": est_in // 4, "output_tokens": len(out_text) // 4, "cost_usd": 0.0, "usage_source": "estimate"}


# ---- one item ----------------------------------------------------------------------------------

def load_fixture(name: str, mods: dict, fixtures_dir: Path = HERE / "fixtures") -> list:
    Page = mods["scraper"].Page
    return [Page(p["url"], p["title"], p["text"]) for p in json.loads((fixtures_dir / f"{name}.json").read_text())]


def retrieve(index, row: dict, translate=None):
    """Returns (search result or None, passages, matched)."""
    if hasattr(index, "search"):
        kwargs = {}
        if row.get("prev"):
            kwargs["prev_question"] = row["prev"]["question"]
        if translate is not None:
            kwargs["translate"] = translate
        found = index.search(row["question"], **kwargs)
        return found, list(found.passages), bool(found.matched)
    passages = index.top(row["question"], 6)  # the old app flow
    return None, passages, bool(passages)


def is_no_answer(text: str, not_found: str = "") -> bool:
    low = text.lower()
    return bool((not_found and not_found.lower() in low) or any(m in low for m in NO_ANSWER_MARKERS))


def score(row: dict, text: str, status: str, not_found: str = "") -> tuple[bool, bool]:
    """(correct, refused). Answerable: an expected phrase and no refusal. Unanswerable: an honest no-answer."""
    refused = status == "no_match" or is_no_answer(text, not_found)
    if not row["answerable"]:
        return refused, refused
    return (not refused) and any(k.lower() in text.lower() for k in row["expect_any"]), refused


def rescore(tag: str, rows: list[dict], out: Path) -> dict:
    """Re-apply the scoring rules to a saved full run (no model calls)."""
    report = json.loads((out / f"{tag}.json").read_text())
    texts = {r["id"]: r["text"] for r in map(json.loads, (out / f"{tag}.raw").read_text().splitlines())}
    by_id = {r["id"]: r for r in rows}
    for item in report["items"]:
        item["correct"], item["refused"] = score(by_id[item["id"]], texts[item["id"]], item["status"])
    report["summary"] = summarize(report["items"], report["mode"])
    (out / f"{tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return report


def answer_item(mods: dict, site: str, row: dict, found, passages, spy: UsageSpy | None, stream=None) -> dict:
    answer, llm = mods["answer"], mods["llm"]
    history = []
    if row.get("prev"):
        history = [{"role": "user", "content": row["prev"]["question"]}, {"role": "assistant", "content": row["prev"]["answer"]}]
    if spy is not None:
        spy.reset()
    kw = {"stream": stream} if stream is not None else {}
    chunks, first, status, result = [], None, "answered", None
    t0 = time.perf_counter()
    try:
        if found is not None and hasattr(answer, "ask"):
            run = answer.ask(site, row["question"], found, history, source=row["fixture"], session="eval", ip="127.0.0.1",
                             cache=None, budget=None, **kw)
        else:
            run = answer.stream_answer(site, row["question"], passages, history, **kw)
        for piece in run:
            if piece and first is None:
                first = time.perf_counter() - t0
            chunks.append(piece)
        result = getattr(run, "result", None)
    except (getattr(answer, "AnswerError", Exception), llm.LLMError) as err:
        status, chunks = "error", [f"ERROR: {type(err).__name__}"]
    total = time.perf_counter() - t0
    text = result.text if result is not None else "".join(chunks)
    if result is not None:
        status = result.status
    usage = getattr(result, "usage", None)
    calls = usage.calls if usage is not None else (spy.posts if spy is not None else (0 if status == "error" else 1))
    est_in = spy.prompt_chars if spy is not None and spy.prompt_chars else sum(len(p.text) for p in passages) + len(row["question"])
    fields = _usage_fields(usage, spy, est_in, text) if calls else {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "usage_source": "none"}
    correct, refused = score(row, text, status, getattr(answer, "NOT_FOUND_TEXT", ""))
    return {"status": status, "correct": correct, "refused": refused, "calls": calls,
            "ttft_s": round(first, 3) if first is not None else None, "total_s": round(total, 3), **fields, "text": text}


# ---- the run -----------------------------------------------------------------------------------

def group(row: dict) -> str:
    if row.get("prev"):
        return "followup"
    return "multilingual" if row["lang"] != "en" else "single"


def evaluate(code_dir: Path, mode: str, rows: list[dict], fixtures_dir: Path = HERE / "fixtures", stream=None, use_translate: bool = True) -> dict:
    mods = load_code(code_dir)
    llm = mods["llm"]
    spy = None
    if mode == "full" and stream is None:
        if not llm.available():
            raise SystemExit("full mode needs OPENROUTER_API_KEY in the environment")
        spy = UsageSpy(llm.requests)
        llm.requests = spy
    translate_usage = []
    translate = None
    if mode == "full" and stream is None and use_translate and hasattr(llm, "translate_query"):
        def translate(text, lang):
            return llm.translate_query(text, lang, on_usage=translate_usage.append)
    indexes, items = {}, []
    for row in rows:
        if row["fixture"] not in indexes:
            pages = load_fixture(row["fixture"], mods, fixtures_dir)
            indexes[row["fixture"]] = (mods["search"].Index(pages), urlparse(pages[0].url).netloc)
        index, site = indexes[row["fixture"]]
        t0 = time.perf_counter()
        found, passages, matched = retrieve(index, row, translate)
        item = {"id": row["id"], "fixture": row["fixture"], "group": group(row), "lang": row["lang"], "answerable": row["answerable"],
                "retrieval_ms": round((time.perf_counter() - t0) * 1000, 2), "passages": len(passages), "matched": matched,
                "hit4": bool(set(row["expect_urls"]) & {p.url for p in passages[:4]}) if row["answerable"] else None,
                "no_match_ok": (not matched) if not row["answerable"] else None}
        if mode == "full":
            item.update(answer_item(mods, site, row, found, passages, spy, stream))
        items.append(item)
    return {"mode": mode, "code_dir": str(Path(code_dir).resolve()), "rev": git_rev(Path(code_dir)), "date": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
            "api_search": hasattr(mods["search"].Index, "search"), "api_ask": hasattr(mods["answer"], "ask"),
            "summary": summarize(items, mode), "items": items}


def _pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"


def summarize(items: list[dict], mode: str) -> dict:
    ans = [i for i in items if i["answerable"]]
    unans = [i for i in items if not i["answerable"]]
    s = {"items": len(items), "answerable": len(ans), "unanswerable": len(unans),
         "hit4": _pct(sum(i["hit4"] for i in ans), len(ans)),
         "no_match_retrieval": _pct(sum(i["no_match_ok"] for i in unans), len(unans)),
         "avg_passages": round(statistics.mean(i["passages"] for i in items), 2) if items else 0,
         "hit4_by_group": {g: _pct(sum(i["hit4"] for i in ans if i["group"] == g), sum(1 for i in ans if i["group"] == g)) for g in ("single", "followup", "multilingual")}}
    if mode == "full":
        called = [i for i in items if i["calls"]]
        ttft = [i["ttft_s"] for i in called if i["ttft_s"] is not None]
        s.update({
            "correct": _pct(sum(i["correct"] for i in items), len(items)),
            "answerable_correct": _pct(sum(i["correct"] for i in ans), len(ans)),
            "honest_no_answer": _pct(sum(i["correct"] for i in unans), len(unans)),
            "false_refusals": sum(1 for i in ans if i["refused"]),
            "correct_by_group": {g: _pct(sum(i["correct"] for i in items if i["group"] == g), sum(1 for i in items if i["group"] == g)) for g in ("single", "followup", "multilingual")},
            "model_calls": sum(i["calls"] for i in items),
            "model_calls_on_unanswerable": sum(i["calls"] for i in unans),
            "input_tokens": sum(i["input_tokens"] for i in items),
            "output_tokens": sum(i["output_tokens"] for i in items),
            "avg_input_tokens": round(statistics.mean(i["input_tokens"] for i in called)) if called else 0,
            "avg_output_tokens": round(statistics.mean(i["output_tokens"] for i in called)) if called else 0,
            "avg_input_tokens_followup": round(statistics.mean([i["input_tokens"] for i in called if i["group"] == "followup"] or [0])),
            "cost_usd": round(sum(i["cost_usd"] for i in items), 5),
            "ttft_median_s": round(statistics.median(ttft), 2) if ttft else None,
            "ttft_max_s": round(max(ttft), 2) if ttft else None,
            "total_median_s": round(statistics.median(i["total_s"] for i in called), 2) if called else None,
            "errors": sum(1 for i in items if i["status"] == "error"),
            "usage_sources": sorted({i["usage_source"] for i in called}),
        })
    return s


def print_table(report: dict) -> None:
    full = report["mode"] == "full"
    head = f"{'id':<15}{'grp':<13}{'hit4':<6}{'pass':<5}"
    if full:
        head += f"{'ok':<4}{'calls':<6}{'in':>6}{'out':>6}{'ttft':>7}{'total':>7}  status"
    print(head)
    for i in report["items"]:
        hit = "-" if i["hit4"] is None else ("yes" if i["hit4"] else "NO")
        line = f"{i['id']:<15}{i['group']:<13}{hit:<6}{i['passages']:<5}"
        if full:
            ttft = f"{i['ttft_s']:.2f}" if i["ttft_s"] is not None else "-"
            line += f"{'yes' if i['correct'] else 'NO':<4}{i['calls']:<6}{i['input_tokens']:>6}{i['output_tokens']:>6}{ttft:>7}{i['total_s']:>7.2f}  {i['status']}"
        print(line)
    print(f"\n{report['mode']} | code {report['rev']} | search() {report['api_search']} | ask() {report['api_ask']}")
    for k, v in report["summary"].items():
        print(f"  {k:<28}{v}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("retrieval", "full"), default="retrieval")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--code-dir", "--repo", dest="code_dir", default=str(REPO))
    ap.add_argument("--questions", default=str(HERE / "questions.jsonl"))
    ap.add_argument("--only", default="", help="comma-separated question ids")
    ap.add_argument("--no-translate", action="store_true", help="do not pass llm.translate_query to Index.search")
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    ap.add_argument("--rescore", action="store_true", help="re-score results/<tag>.json from <tag>.raw without model calls")
    args = ap.parse_args(argv)
    rows = [json.loads(line) for line in Path(args.questions).read_text().splitlines() if line.strip()]
    if args.only:
        keep = set(args.only.split(","))
        rows = [r for r in rows if r["id"] in keep]
    if args.rescore:
        print_table(rescore(args.tag, rows, Path(args.out_dir)))
        return 0
    report = evaluate(Path(args.code_dir), args.mode, rows, use_translate=not args.no_translate)
    report["tag"] = args.tag
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = [{"id": i["id"], "text": i.pop("text")} for i in report["items"] if "text" in i]
    (out / f"{args.tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    if raw:
        (out / f"{args.tag}.raw").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in raw) + "\n")
    print_table(report)
    print(f"\nwrote {out / (args.tag + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
