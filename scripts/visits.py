#!/usr/bin/env python3
"""Who visited slimanechanchoul.com: the newest page views and the totals from the API's visit log (GET /api/stats),
with the time each visitor spent on the page and what they clicked (from hit.js's leave summary; blank when the
visitor is still there, left within half a second, or the API is older than the summaries).

  python3 scripts/visits.py                              # last 7 days, newest 30 visits
  python3 scripts/visits.py --days 30 --rows 60
  python3 scripts/visits.py --api http://127.0.0.1:8090  # a local API
  python3 scripts/visits.py --json                       # the raw JSON

The key is read from ~/work/.env (line stats_key=...) or from the STATS_KEY environment variable; it is never
printed and only sent over https (plain http only to localhost). Times are UTC. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_API = "https://site-chatbot-omega.vercel.app"
ENV_FILE = Path.home() / "work" / ".env"
_CTRL = re.compile(r"[\x00-\x1f\x7f-\x9f​-‏ -‮⁦-⁩﻿]")  # no terminal escapes


class StatsError(Exception):
    pass


def read_key(env_file: Path = ENV_FILE, environ=None) -> str:
    """STATS_KEY from the environment, else the stats_key=... line of the .env file; "" when neither is set."""
    environ = os.environ if environ is None else environ
    v = (environ.get("STATS_KEY") or "").strip().strip("'\"").strip()
    if v:
        return v
    try:
        lines = Path(env_file).read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in lines:
        name, sep, value = line.partition("=")
        name = name.strip()
        if name.startswith("export "):
            name = name[len("export "):].strip()
        if sep and name.lower() == "stats_key":
            return value.strip().strip("'\"").strip()
    return ""


def _safe_api(api: str) -> str:
    u = urllib.parse.urlsplit(api.strip())
    local = u.hostname in ("localhost", "127.0.0.1", "::1")
    if u.scheme != "https" and not (u.scheme == "http" and local):
        raise StatsError("the API address must start with https:// (plain http only for localhost)")
    return api.strip().rstrip("/")


def fetch(api: str, key: str, days: int, limit: int, timeout: float = 20.0) -> dict:
    url = _safe_api(api) + "/api/stats?" + urllib.parse.urlencode({"days": days, "limit": limit})
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "visits-cli/1"})
    req.add_unredirected_header("X-Stats-Key", key)  # never forwarded to a redirect target
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise StatsError({
            401: "the API refused the key: stats_key in ~/work/.env must equal STATS_KEY on the API",
            404: "no visit stats on this API: STATS_KEY is not set there, or this version has no /api/stats",
            503: "the API cannot read its visit store right now (check the Upstash connection)",
        }.get(e.code, f"the API answered HTTP {e.code}")) from None
    except (urllib.error.URLError, OSError) as e:
        raise StatsError(f"cannot reach {api}: {getattr(e, 'reason', e)}") from None
    except ValueError:
        raise StatsError("the API did not answer JSON") from None


def clean(v, n: int) -> str:
    s = _CTRL.sub(" ", "" if v is None else str(v)).strip()
    return s if len(s) <= n else s[:max(1, n - 1)] + "…"


def _table(headers: list[str], rows: list[list], widths: list[int], notes: list[list[str]] | None = None) -> list[str]:
    """Rows of fixed-width columns; notes[i] = lines printed under row i."""
    out = ["  " + "  ".join(h.ljust(w) for h, w in zip(headers, widths)).rstrip()]
    for i, r in enumerate(rows):
        out.append("  " + "  ".join(clean(c, w).ljust(w) for c, w in zip(r, widths)).rstrip())
        out += (notes[i] if notes and i < len(notes) else [])
    return out


def _ms(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0 else None


def duration(ms) -> str:
    """Time on page: "45s", "3m 05s", "1h 02m"; "" when unknown."""
    if _ms(ms) is None:
        return ""
    s = int(round(ms / 1000))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    return f"{m}m {s:02d}s" if m < 60 else f"{m // 60}h {m % 60:02d}m"


def _at(ms) -> str:
    """When a click happened, from the page load: "0:07", "12:40"."""
    s = int(round((_ms(ms) or 0) / 1000))
    return f"{s // 60}:{s % 60:02d}"


def click_lines(clicks, indent: int = 6, width: int = 110) -> list[str]:
    """Clicks of one visit, in order: "Imagery 0:07, Topo 0:09, Zoom in x3 0:15" (x3 = clicked 3 times in a row)."""
    items = []
    for c in clicks if isinstance(clicks, list) else []:
        if isinstance(c, dict) and c.get("l"):
            n = c.get("n")
            rep = f" x{clean(n, 5)}" if isinstance(n, int) and not isinstance(n, bool) and n > 1 else ""
            items.append(f"{clean(c.get('l'), 40)}{rep} {_at(c.get('ms'))}")
    if not items:
        return []
    pad = " " * indent
    return textwrap.wrap(", ".join(items), width, initial_indent=pad + "clicks: ", subsequent_indent=pad + "        ")


def render(data: dict, rows: int = 30, top: int = 10) -> str:
    lines = [f"Visits {data.get('from')} to {data.get('to')} (UTC): {data.get('views', 0)} page views, "
             f"{data.get('sessions', 0)} sessions (store: {clean(data.get('store'), 10)})"]
    if data.get("partial"):
        lines.append("Note: the log keeps the last 5,000 visits, so city, referrer and session counts may miss older ones.")
    evs = (data.get("events") or [])[:max(0, rows)]
    lines.append("")
    if evs:
        lines.append(f"Newest {len(evs)} visits (Time = time on page while the tab was in front)")
        lines += _table(
            ["UTC time", "Session", "CC", "City", "Page", "Time", "From", "Device"],
            [[f"{str(e.get('utc', ''))[5:10]} {str(e.get('utc', ''))[11:19]}", str(e.get("sid") or "")[:4],
              e.get("country") or "??", e.get("city") or "", f"{e.get('path', '')}{e.get('qs', '')}",
              duration(e.get("active_ms")), e.get("ref") or "(direct)", f"{e.get('device', '')} {e.get('browser', '')}"]
             for e in evs],
            [14, 7, 2, 16, 30, 7, 24, 15],
            [click_lines(e.get("clicks")) for e in evs])
    else:
        lines.append("No visits in this period.")
    eng = [p for p in data.get("engagement") or [] if isinstance(p, dict)][:top]
    if eng:
        lines += ["", "Time on page (median) and most clicked, per page"]
        for p in eng:
            n = p.get("n") if isinstance(p.get("n"), int) else 0
            lines.append(f"  {duration(p.get('median_ms')).rjust(7)}  {clean(p.get('key'), 60)}  "
                         f"({n} visit{'' if n == 1 else 's'}, {clean(p.get('clicks', 0), 6)} clicks)")
            tops = [f"{clean(t.get('key'), 40)} {clean(t.get('n'), 5)}" for t in p.get("top") or [] if isinstance(t, dict)]
            if tops:
                lines += textwrap.wrap(", ".join(tops), 110, initial_indent="           top: ",
                                       subsequent_indent="                ")
    days = [f"{str(d.get('day', ''))[5:]} {d.get('views', 0)}" for d in data.get("by_day") or []]
    if days:
        lines += ["", "By day (views)"] + textwrap.wrap(", ".join(days), 100, initial_indent="  ",
                                                         subsequent_indent="  ")
    for title, name in (("By country", "by_country"), ("By page", "by_page"), ("By referrer", "by_ref"),
                        ("By city", "by_city")):
        items = data.get(name) or []
        if items:
            lines += ["", title] + [f"  {clean(it.get('n'), 6).rjust(6)}  {clean(it.get('key'), 70)}"
                                    for it in items[:top]]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Visits to slimanechanchoul.com from the API's visit log (UTC).")
    ap.add_argument("--days", type=int, default=7, help="period in UTC days, today included (default 7)")
    ap.add_argument("--rows", type=int, default=30, help="newest visits to list (default 30)")
    ap.add_argument("--api", default=DEFAULT_API, help=f"API base URL (default {DEFAULT_API})")
    ap.add_argument("--env", default=str(ENV_FILE), help="file with the stats_key=... line (default ~/work/.env)")
    ap.add_argument("--json", action="store_true", help="print the raw JSON")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    key = read_key(Path(a.env).expanduser())
    if not key:
        print("No stats key: add a line stats_key=... to ~/work/.env (the same value as STATS_KEY on the API).",
              file=sys.stderr)
        return 2
    try:
        data = fetch(a.api, key, a.days, max(0, a.rows))
    except StatsError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(data, ensure_ascii=False, indent=1) if a.json else render(data, a.rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
