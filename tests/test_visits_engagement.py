"""Time on page and clicks: the leave summary hit.js sends once per page view (POST /api/hit type "leave"), its
join with the page views in GET /api/stats, and scripts/visits.py. Backward compatibility both ways: page-view bodies
(old hit.js) behave exactly as before, and a summary from the new hit.js is refused (never counted as a page view)
by an API from before this change."""
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import visits  # noqa: E402
from test_api import Clock  # noqa: E402
from test_visits import CHROME, IP, SITE, FakeUpstash, _cli, body, hit, make, stored  # noqa: E402

KEY = {"X-Stats-Key": "s3cret-key"}
T0 = 1_800_000_000_123  # the page-load time hit.js sends as t (page view) and pv (summary)


def leave(**kw):
    b = {"type": "leave", "page": "/forest-match/", "sid": "a1b2c3d4e5", "pv": T0, "active_ms": 8123,
         "max_scroll_pct": 64, "clicks": [{"l": "Imagery", "ms": 2100}, {"l": "Topo", "ms": 3900},
                                          {"l": "Similarity score, same forest type", "ms": 5200}]}
    b.update(kw)
    return json.dumps(b)


def leaves(store) -> list[dict]:
    return [json.loads(x) for x in store.lrange(0, 10_000, key=visits.LEAVE_KEY)]


# ---------------------------------------------------------------- POST /api/hit type "leave"

def test_leave_is_stored_apart_with_no_ip_geo_or_user_agent(monkeypatch):
    c, store, _ = make(monkeypatch)
    r = hit(c, leave(), **{"x-vercel-ip-country": "US", "x-vercel-ip-city": "Denver"})
    assert r.status_code == 204 and r.content == b"" and r.headers["cache-control"] == "no-store"
    assert stored(store) == []                                  # not a page view
    assert store.day_hashes(["2027-01-15"]) == [{}]              # the daily page-view counters do not move
    [lv] = leaves(store)
    assert lv == {"type": "leave", "ts": 1_800_000_000, "utc": "2027-01-15T08:00:00Z", "path": "/forest-match/",
                  "sid": "a1b2c3d4e5", "pv": T0, "active_ms": 8123, "scroll": 64,
                  "clicks": [{"l": "Imagery", "ms": 2100}, {"l": "Topo", "ms": 3900},
                             {"l": "Similarity score, same forest type", "ms": 5200}]}
    raw = store.lrange(0, 0, key=visits.LEAVE_KEY)[0]
    assert IP not in raw and "Denver" not in raw and "US" not in raw.replace("\"utc\"", "")
    assert "Mozilla" not in raw and "Chrome" not in raw and "Windows" not in raw


def test_old_page_view_bodies_behave_exactly_as_before(monkeypatch):
    c, store, _ = make(monkeypatch)
    old = json.loads(body())  # what the live hit.js sends: no "type"
    assert hit(c, json.dumps(old)).status_code == 204
    expect = visits.make_event(old, {}, CHROME, 1_800_000_000)
    assert stored(store) == [expect] and "type" not in expect and "active_ms" not in expect
    for other in ("view", "LEAVE", "", None, 7):  # anything but "leave" is a page view, as the old API read it
        assert hit(c, json.dumps({**old, "type": other})).status_code == 204
    assert len(stored(store)) == 6 and leaves(store) == []
    assert store.day_hashes(["2027-01-15"])[0]["p:/ask/"] == 6


def test_a_summary_from_the_new_hitjs_is_refused_by_an_old_api():
    # make_event is the page-view parser the API had before summaries existed (unchanged): without a "path" the
    # summary is a 400 there, so a new hit.js live before the new API never adds fake page views.
    assert visits.make_event(json.loads(leave()), {}, CHROME, 1_800_000_000) is None
    assert "path" not in json.loads(leave())


def test_leave_accepts_path_as_the_page_too(monkeypatch):
    c, store, _ = make(monkeypatch)
    b = json.loads(leave())
    b["path"] = b.pop("page")
    assert hit(c, json.dumps(b)).status_code == 204
    assert leaves(store)[0]["path"] == "/forest-match/" and stored(store) == []


@pytest.mark.parametrize("payload", [
    json.dumps({"type": "leave"}),
    leave(page="forest-match"),             # no leading slash
    leave(page=None),
    leave(page=["/a/"]),
    leave(sid=""),
    leave(sid="bad sid!"),
    leave(sid=None),
    leave(clicks="Imagery"),                 # clicks must be a list
    leave(clicks={"l": "Imagery"}),
    leave(clicks=5),
    "[" * 1500,                              # nested too deep to parse: 400, not 500
    '{"type": "leave", "page": "/a/", "sid": "abcdefgh", "clicks": [',
])
def test_malformed_leave_bodies_are_refused(monkeypatch, payload):
    c, store, _ = make(monkeypatch)
    r = hit(c, payload)
    assert r.status_code == 400 and r.json()["error"]["code"] == "bad_request"
    assert leaves(store) == [] and stored(store) == []


def test_leave_numbers_are_brought_into_range_and_bad_clicks_dropped(monkeypatch):
    c, store, _ = make(monkeypatch)
    clicks = [{"l": "ok", "ms": -5}, "Imagery", None, {"ms": 10}, {"l": 42}, {"l": "  \x00\u200b  "},
              {"l": "Zoom in", "ms": 10**15, "n": 3}, {"l": "x", "n": True}, {"l": "y", "n": 10**9},
              {"l": "z", "ms": "soon", "n": -2}]
    for active, scroll, want in ((-1, -9, (0, 0)), (10**12, 250, (visits.MAX_ACTIVE_MS, 100)),
                                 ("long", True, (0, 0)), (None, 99.7, (0, 99))):
        assert hit(c, leave(active_ms=active, max_scroll_pct=scroll, clicks=clicks)).status_code == 204
        lv = leaves(store)[0]
        assert (lv["active_ms"], lv["scroll"]) == want
        assert lv["clicks"] == [{"l": "ok", "ms": 0}, {"l": "Zoom in", "ms": visits.MAX_CLICK_MS, "n": 3},
                                {"l": "x", "ms": 0}, {"l": "y", "ms": 0, "n": visits.MAX_REPEAT}, {"l": "z", "ms": 0}]
    r = hit(c, '{"type": "leave", "page": "/a/", "sid": "abcdefgh", "active_ms": NaN, "pv": Infinity}')
    assert r.status_code == 204 and leaves(store)[0]["active_ms"] == 0 and leaves(store)[0]["pv"] == 0
    assert hit(c, leave(clicks=None)).status_code == 204 and leaves(store)[0]["clicks"] == []
    assert hit(c, leave(pv="x")).status_code == 204 and leaves(store)[0]["pv"] == 0


def test_leave_caps_body_clicks_and_labels(monkeypatch):
    c, store, _ = make(monkeypatch)
    many = [{"l": f"Button {i}", "ms": i} for i in range(30)]
    assert hit(c, leave(clicks=many)).status_code == 204
    assert [x["l"] for x in leaves(store)[0]["clicks"]] == [f"Button {i}" for i in range(visits.MAX_CLICKS)]
    assert hit(c, leave(clicks=[{"l": "L" * 100, "ms": 1}])).status_code == 204
    assert leaves(store)[0]["clicks"][0]["l"] == "L" * visits.MAX_LABEL
    # a full summary in Arabic (25 labels of 40 characters, 2 bytes each) passes the page view's 2 KB, fits the 4 KB
    full = leave(clicks=[{"l": f"{i:02d}" + "\u0632\u0631" * 19, "ms": 1_000_000 + i} for i in range(25)],
                 page="/ask/").replace("\\u0632\\u0631", "\u0632\u0631")
    assert visits.MAX_BODY < len(full.encode()) <= visits.MAX_LEAVE_BODY
    assert hit(c, full).status_code == 204 and len(leaves(store)[0]["clicks"]) == 25
    assert leaves(store)[0]["clicks"][0]["l"] == "00" + "\u0632\u0631" * 19
    too_big = leave(clicks=[{"l": "w" * 40, "ms": 1, "pad": "p" * 200} for _ in range(25)])
    assert len(too_big.encode()) > visits.MAX_LEAVE_BODY
    assert hit(c, too_big).status_code == 413  # declared Content-Length

    def chunks():  # no Content-Length: counted while it streams
        for _ in range(5):
            yield b"x" * 1000

    r = c.post("/api/hit", content=chunks(), headers={"Origin": SITE, "User-Agent": CHROME, "Content-Type": "text/plain"})
    assert r.status_code == 413
    # a page view between 2 and 4 KB is still too big, as before
    pv = body(title="t" * 100)
    pv = pv[:-1] + ', "pad": "' + "p" * 2500 + '"}'
    assert visits.MAX_BODY < len(pv.encode()) <= visits.MAX_LEAVE_BODY
    assert hit(c, pv).status_code == 413
    assert hit(c, "x" * 3000).status_code == 413  # junk over 2 KB too
    assert len(leaves(store)) == 3 and stored(store) == []


def test_leave_labels_are_cleaned_and_keep_no_addresses(monkeypatch):
    c, store, _ = make(monkeypatch)
    labels = ["Hi\x1b[31m red\u202e evil\n", "Mail me at jane.doe@example.com", "https://x.example/a/b?token=abc#top",
              "see www.example.org/p?q=1 now", "out:Upwork.com.", "out:evil.example/path?x=1", "out:", "out:a b",
              "\ud83c", "Tree \U0001F332 map"]
    assert hit(c, leave(clicks=[{"l": label, "ms": 1} for label in labels])).status_code == 204
    got = [x["l"] for x in leaves(store)[0]["clicks"]]
    assert got == ["Hi [31m red evil", "Mail me at (email)", "https://x.example/a/b", "see www.example.org/p now",
                   "out:upwork.com", "Tree \U0001F332 map"]
    raw = store.lrange(0, 0, key=visits.LEAVE_KEY)[0]
    assert "token" not in raw and "jane" not in raw and "q=1" not in raw and "path" not in raw.replace('"path"', "")
    assert visits.clean_label("out:" + "a" * 60 + ".com") == ("out:" + "a" * 60)[:visits.MAX_LABEL]


def test_a_lone_surrogate_cannot_break_the_stats(monkeypatch):
    c, store, _ = make(monkeypatch)  # a title or label cut in the middle of an emoji by a client
    assert hit(c, body(title="Forest \ud83c")).status_code == 204
    assert hit(c, leave(page="/ask/", clicks=[{"l": "Map \ud83c", "ms": 1}])).status_code == 204
    r = c.get("/api/stats", headers=KEY)
    assert r.status_code == 200
    assert r.json()["events"][0]["title"] == "Forest" and r.json()["engagement"][0]["top"][0]["key"] == "Map"


def test_leave_follows_the_origin_bot_and_cap_rules(monkeypatch):
    clock = Clock()
    c, store, _ = make(monkeypatch, clock=clock, HIT_IP_CAP=2, HIT_DAILY_CAP=3)
    assert hit(c, leave(), origin=None).status_code == 403
    assert hit(c, leave(), origin="https://evil.example").status_code == 403
    assert hit(c, leave(), ua="Mozilla/5.0 (X11; Linux x86_64) HeadlessChrome/129.0 Safari/537.36").status_code == 204
    assert leaves(store) == []
    # a summary counts in the same caps as page views
    assert [hit(c, body(), ip="198.51.100.1").status_code, hit(c, leave(), ip="198.51.100.1").status_code,
            hit(c, leave(), ip="198.51.100.1").status_code] == [204, 204, 429]
    assert hit(c, leave(), ip="198.51.100.2").status_code == 204
    r = hit(c, body(), ip="198.51.100.3")
    assert r.status_code == 429 and r.json()["error"]["code"] == "hit_cap"
    assert len(stored(store)) == 1 and len(leaves(store)) == 2
    assert hit(c, leave(sid=""), ip="198.51.100.9").status_code == 400  # refused before the cap counts it


def test_upstash_leave_is_two_commands_and_the_quota_bound_holds(monkeypatch):
    fake = FakeUpstash()
    c, store, app = make(monkeypatch, store=visits.UpstashStore("https://x.upstash.io", "tok", post=fake,
                                                                max_events=3))
    for i in range(5):
        assert hit(c, leave(pv=T0 + i), ip=f"198.51.100.{i}").status_code == 204
    writes = [call["json"] for call in fake.calls]
    assert writes[0][0][:2] == ["LPUSH", "visits:leave"] and writes[0][1] == ["LTRIM", "visits:leave", "0", "2"]
    assert all(len(w) == 2 for w in writes)  # no daily counters, no EXPIRE
    assert [json.loads(x)["pv"] for x in store.lrange(0, 99, key=visits.LEAVE_KEY)] == [T0 + 4, T0 + 3, T0 + 2]
    assert store.lrange(0, 99) == []  # the page-view list is untouched
    caps = app.state.hit_caps
    assert (caps.daily_cap * 4 + 1) * 31 < 500_000  # a capped request is 4 commands at most (a page view)


def test_a_broken_store_never_breaks_a_leave(monkeypatch):
    class Boom(visits.MemoryStore):
        def add_leave(self, event):
            raise RuntimeError("store exploded")

    c, _, _ = make(monkeypatch, store=Boom())
    assert hit(c, leave()).status_code == 204
    c, _, _ = make(monkeypatch, store=visits.UpstashStore("https://x.upstash.io", "tok", post=FakeUpstash(fail="raise")))
    assert hit(c, leave()).status_code == 204


# ---------------------------------------------------------------- GET /api/stats: the join

def _log(monkeypatch, store=None):
    clock = Clock()  # 2027-01-15 08:00 UTC
    c, store, _ = make(monkeypatch, store=store, clock=clock)
    hit(c, body(path="/forest-match/", sid="s1", t=T0))                       # view 1
    clock.t += 9
    hit(c, leave(sid="s1", pv=T0, active_ms=8200))                              # its summary
    clock.t += 30
    hit(c, body(path="/forest-match/", sid="s1", t=T0 + 40_000))              # view 2: same tab, reloaded
    clock.t += 5
    hit(c, body(path="/ask/", sid="s2", t=T0 + 45_000))                        # view 3: no summary yet
    clock.t += 60
    hit(c, leave(sid="s1", pv=T0 + 40_000, active_ms=61_000, max_scroll_pct=100,
                 clicks=[{"l": "Topo", "ms": 1000}, {"l": "Zoom in", "ms": 2000, "n": 4}]))
    hit(c, leave(sid="s1", pv=T0 + 40_000, active_ms=99_999))                 # sent twice: counts once
    hit(c, leave(page="/other/", sid="s9", pv=T0, active_ms=5000, clicks=[{"l": "out:upwork.com", "ms": 4000}]))
    return c, store


@pytest.mark.parametrize("kind", ["memory", "upstash"])
def test_stats_gives_each_view_its_time_and_clicks(monkeypatch, kind):
    store = visits.MemoryStore() if kind == "memory" else visits.UpstashStore("https://x.upstash.io", "tok",
                                                                               post=FakeUpstash())
    c, store = _log(monkeypatch, store)
    d = c.get("/api/stats?days=1", headers=KEY).json()
    assert d["views"] == 3 and d["events_in_period"] == 3 and d["leaves_in_period"] == 3 and not d["leaves_partial"]
    ask, fm2, fm1 = d["events"]  # newest first
    assert (ask["path"], ask["active_ms"], ask["scroll"], ask["clicks"]) == ("/ask/", None, None, [])
    assert (fm2["active_ms"], fm2["scroll"]) == (61_000, 100)
    assert fm2["clicks"] == [{"l": "Topo", "ms": 1000}, {"l": "Zoom in", "ms": 2000, "n": 4}]
    assert (fm1["active_ms"], fm1["scroll"], [x["l"] for x in fm1["clicks"]]) == \
        (8200, 64, ["Imagery", "Topo", "Similarity score, same forest type"])
    eng = {p["key"]: p for p in d["engagement"]}
    assert list(eng) == ["/forest-match/", "/other/"]
    assert eng["/forest-match/"] == {"key": "/forest-match/", "n": 2, "median_ms": 34_600, "clicks": 8,
                                     "top": [{"key": "Zoom in", "n": 4}, {"key": "Topo", "n": 2},
                                             {"key": "Imagery", "n": 1},
                                             {"key": "Similarity score, same forest type", "n": 1}]}
    assert eng["/other/"]["top"] == [{"key": "out:upwork.com", "n": 1}] and eng["/other/"]["median_ms"] == 5000
    assert d["by_page"] == [{"key": "/forest-match/", "n": 2}, {"key": "/ask/", "n": 1}]  # summaries are not views
    assert d["sessions"] == 2 and IP not in json.dumps(d)


def test_join_without_pv_pairs_the_oldest_view_after_it():
    views = [{"ts": 100, "sid": "s", "path": "/a/", "ct": 0}, {"ts": 200, "sid": "s", "path": "/a/", "ct": 0},
             {"ts": 150, "sid": "", "path": "/a/"}]
    lvs = [{"ts": 260, "sid": "s", "path": "/a/", "pv": 0, "active_ms": 2},
           {"ts": 120, "sid": "s", "path": "/a/", "pv": 0, "active_ms": 1},
           {"ts": 130, "sid": "s", "path": "/b/", "pv": 0, "active_ms": 9}, {"ts": "x", "sid": 5, "path": None}]
    kept = visits.join_leaves(views, lvs)
    assert [v["active_ms"] for v in views] == [1, 2, None]
    assert len(kept) == 3
    assert visits.engagement(kept)[0] == {"key": "/a/", "n": 2, "median_ms": 1, "clicks": 0, "top": []}


def test_stats_still_answers_when_the_summaries_cannot_be_read(monkeypatch):
    class NoLeaves(visits.MemoryStore):
        def lrange(self, start, stop, key=visits.EVENTS_KEY):
            return None if key == visits.LEAVE_KEY else super().lrange(start, stop)

    class OldStore(visits.MemoryStore):  # a store without the summary list at all
        def lrange(self, start, stop):
            return super().lrange(start, stop)

    for store in (NoLeaves(), OldStore()):
        c, store, _ = make(monkeypatch, store=store)
        assert hit(c, body()).status_code == 204
        r = c.get("/api/stats", headers=KEY)
        assert r.status_code == 200
        d = r.json()
        assert d["views"] == 1 and d["leaves_in_period"] is None and d["engagement"] == []
        assert d["events"][0]["active_ms"] is None and d["events"][0]["clicks"] == []


# ---------------------------------------------------------------- scripts/visits.py

def test_cli_shows_time_on_page_and_clicks_per_visit(monkeypatch, tmp_path, capsys):
    cli = _cli()
    monkeypatch.delenv("STATS_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("stats_key=k\n")
    ev = {"utc": "2027-01-15T08:00:00Z", "sid": "abcd1234", "country": "US", "city": "Denver",
          "path": "/forest-match/", "qs": "", "ref": "www.upwork.com", "device": "desktop", "browser": "Chrome"}
    data = {"store": "upstash", "from": "2027-01-15", "to": "2027-01-15", "views": 2, "sessions": 1,
            "events": [{**ev, "active_ms": 83_400, "scroll": 100,
                        "clicks": [{"l": "Imagery", "ms": 2100}, {"l": "Zoom in", "ms": 7000, "n": 3},
                                   {"l": "\x1b[2Jboom", "ms": 65_000}]},
                       {**ev, "path": "/ask/", "active_ms": None, "clicks": []}],
            "engagement": [{"key": "/forest-match/", "n": 1, "median_ms": 83_400, "clicks": 5,
                            "top": [{"key": "Zoom in", "n": 3}, {"key": "Imagery", "n": 1}]}]}
    monkeypatch.setattr(cli, "fetch", lambda api, key, days, limit: data)
    assert cli.main(["--env", str(env)]) == 0
    out = capsys.readouterr().out
    assert "Time" in out and "1m 23s" in out
    assert "clicks: Imagery 0:02, Zoom in x3 0:07, [2Jboom 1:05" in out and "\x1b" not in out
    assert "Time on page (median)" in out and "1m 23s  /forest-match/  (1 visit, 5 clicks)" in out
    assert "top: Zoom in 3, Imagery 1" in out
    assert [cli.duration(x) for x in (None, -1, "x", 0, 8_200, 59_600, 3_725_000)] == \
        ["", "", "", "0s", "8s", "1m 00s", "1h 02m"]


def test_cli_still_reads_an_api_without_summaries(monkeypatch, tmp_path, capsys):
    cli = _cli()
    monkeypatch.delenv("STATS_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("stats_key=k\n")
    data = {"store": "upstash", "from": "2027-01-15", "to": "2027-01-15", "views": 1, "sessions": 1,
            "events": [{"utc": "2027-01-15T08:00:00Z", "sid": "abcd", "path": "/", "ref": ""}]}
    monkeypatch.setattr(cli, "fetch", lambda api, key, days, limit: data)
    assert cli.main(["--env", str(env)]) == 0
    out = capsys.readouterr().out
    assert "Newest 1 visits" in out and "clicks:" not in out and "Time on page" not in out
