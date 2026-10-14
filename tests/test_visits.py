"""Visit log: POST /api/hit, GET /api/stats, the stores in visits.py (Upstash faked, no network) and the
scripts/visits.py CLI."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient  # noqa: E402

import visits  # noqa: E402
from api import main  # noqa: E402
from test_api import Clock, FakeLLM  # noqa: E402

SITE = "https://slimanechanchoul.com"
CHROME = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
          "Chrome/129.0.0.0 Safari/537.36")
IPHONE = ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) "
          "Version/17.6 Mobile/15E148 Safari/604.1")
IP = "203.0.113.77"


class Resp:
    def __init__(self, status, data):
        self.status_code, self._data = status, data

    def json(self):
        if isinstance(self._data, Exception):
            raise self._data
        return self._data


class FakeUpstash:
    """Runs pipeline commands on dicts and answers like Upstash's REST API ([{result}] or [{error}])."""

    def __init__(self, token="tok", fail=None):
        self.token, self.fail, self.calls = token, fail, []
        self.lists, self.hashes, self.ttl = {}, {}, {}

    def __call__(self, url, json=None, timeout=None, headers=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout, "headers": headers})
        if self.fail == "raise":
            raise ConnectionError("store down")
        if self.fail == "500":
            return Resp(500, {"error": "boom"})
        if self.fail == "garbage":
            return Resp(200, ValueError("not json"))
        if (headers or {}).get("Authorization") != "Bearer " + self.token:
            return Resp(401, {"error": "Unauthorized"})
        out = []
        for op, key, *args in json:
            if op == "LPUSH":
                self.lists.setdefault(key, []).insert(0, args[0])
                out.append({"result": len(self.lists[key])})
            elif op == "LTRIM":
                self.lists[key] = self.lists.get(key, [])[int(args[0]):int(args[1]) + 1]
                out.append({"result": "OK"})
            elif op == "HINCRBY":
                h = self.hashes.setdefault(key, {})
                h[args[0]] = h.get(args[0], 0) + int(args[1])
                out.append({"result": h[args[0]]})
            elif op == "EXPIRE":
                self.ttl[key] = int(args[0])
                out.append({"result": 1})
            elif op == "LRANGE":
                out.append({"result": self.lists.get(key, [])[int(args[0]):int(args[1]) + 1]})
            elif op == "HGETALL":
                out.append({"result": [x for f, v in self.hashes.get(key, {}).items() for x in (f, str(v))]})
            else:
                out.append({"error": f"ERR unknown command '{op}'"})
        return Resp(200, out)


def make(monkeypatch, store=None, clock=None, stats_key="s3cret-key", **env):
    for k in ("STATS_KEY", "HIT_IP_CAP", "HIT_DAILY_CAP", "CORS_ORIGINS", "CORS_ORIGIN_REGEX", "CLIENT_IP_HEADER"):
        monkeypatch.delenv(k, raising=False)
    if stats_key:
        monkeypatch.setenv("STATS_KEY", stats_key)
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    store = store if store is not None else visits.MemoryStore()
    app = main.create_app(stream=FakeLLM(), check_key=False, visits=store, clock=clock or Clock())
    return TestClient(app), store, app


def body(**kw):
    b = {"path": "/ask/", "qs": "?site=client.example", "title": "Ask any website", "lang": "en-US",
         "tz": "America/New_York", "sw": 1920, "sh": 1080, "sid": "a1b2c3d4e5", "t": 1_800_000_000_123,
         "ref": "https://www.upwork.com/jobs/~01abc?source=rss#apply"}
    b.update(kw)
    return json.dumps(b)


def hit(c, payload=None, origin=SITE, ua=CHROME, ip=IP, ctype="text/plain;charset=UTF-8", **headers):
    h = {"Content-Type": ctype, "User-Agent": ua, "X-Forwarded-For": ip, **headers}
    if origin is not None:
        h["Origin"] = origin
    return c.post("/api/hit", content=payload if payload is not None else body(), headers=h)


def stored(store) -> list[dict]:
    return [json.loads(x) for x in store.lrange(0, 10_000)]


# ---------------------------------------------------------------- POST /api/hit

def test_hit_stores_a_clean_event_with_geo_and_no_ip_or_user_agent(monkeypatch):
    c, store, _ = make(monkeypatch)
    r = hit(c, **{"x-vercel-ip-country": "br", "x-vercel-ip-country-region": "SP",
                  "x-vercel-ip-city": "S%C3%A3o%20Paulo", "x-vercel-ip-timezone": "America/Sao_Paulo"})
    assert r.status_code == 204 and r.content == b""
    assert r.headers["cache-control"] == "no-store"
    [ev] = stored(store)
    assert ev["path"] == "/ask/" and ev["qs"] == "?site=client.example" and ev["title"] == "Ask any website"
    assert ev["ref"] == "www.upwork.com"  # host only: no path, no query, no fragment
    assert (ev["country"], ev["region"], ev["city"], ev["ip_tz"]) == ("BR", "SP", "São Paulo", "America/Sao_Paulo")
    assert (ev["device"], ev["browser"], ev["os"]) == ("desktop", "Chrome", "Windows")
    assert (ev["lang"], ev["tz"], ev["sw"], ev["sh"], ev["sid"], ev["ct"]) == \
        ("en-US", "America/New_York", 1920, 1080, "a1b2c3d4e5", 1_800_000_000_123)
    assert ev["ts"] == 1_800_000_000 and ev["utc"] == "2027-01-15T08:00:00Z"
    raw = store.lrange(0, 0)[0]
    assert IP not in raw and "testclient" not in raw  # never the IP
    assert "Mozilla" not in raw and "AppleWebKit" not in raw and "129.0" not in raw  # never the user agent
    assert "jobs" not in raw and "source=rss" not in raw and "apply" not in raw


def test_hit_accepts_a_json_content_type_and_fills_unknowns(monkeypatch):
    c, store, _ = make(monkeypatch)
    r = hit(c, json.dumps({"path": "/", "sw": "wide", "t": "soon", "lang": "<script>", "ref": "not a url"}),
            ctype="application/json", ua=IPHONE)
    assert r.status_code == 204
    [ev] = stored(store)
    assert ev["path"] == "/" and ev["country"] == "" and ev["city"] == "" and ev["ref"] == ""
    assert ev["sw"] == 0 and ev["ct"] == 0 and ev["lang"] == "" and ev["qs"] == ""
    assert (ev["device"], ev["browser"], ev["os"]) == ("mobile", "Safari", "iOS")


def test_hit_needs_the_site_origin(monkeypatch):
    c, store, _ = make(monkeypatch)
    for origin in (None, "https://evil.example", "null", "https://slimanechanchoul.com.evil.example"):
        r = hit(c, origin=origin)
        assert r.status_code == 403 and r.json()["error"]["code"] == "origin_not_allowed", origin
    assert stored(store) == []
    assert hit(c, origin="http://localhost:8091").status_code == 204  # a local copy of the site (dev)


def test_hit_drops_bots_and_keeps_real_browsers(monkeypatch):
    c, store, _ = make(monkeypatch)
    bots = ["", "curl/8.5.0", "python-requests/2.32", "Go-http-client/2.0",
            "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
            "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko; compatible; bingbot/2.0) Chrome/116 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/129.0 Safari/537.36",
            "Mozilla/5.0 (Linux; Android 11; moto g power) AppleWebKit/537.36 Chrome/129 Mobile Safari/537.36 "
            "Chrome-Lighthouse",
            "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)",
            "Mozilla/5.0 (compatible; SemrushBot/7~bl; +http://www.semrush.com/bot.html)",
            "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko); compatible; ChatGPT-User/1.0",
            "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 Chrome/120 Safari/537.36 PTST/240101 Prerender"]
    for ua in bots:
        assert hit(c, ua=ua).status_code == 204, ua
    assert stored(store) == []
    people = {
        IPHONE: ("mobile", "Safari", "iOS"),
        CHROME: ("desktop", "Chrome", "Windows"),
        "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Mobile "
        "Safari/537.36": ("mobile", "Chrome", "Android"),
        "Mozilla/5.0 (Linux; Android 10; CUBOT X30) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Mobile "
        "Safari/537.36": ("mobile", "Chrome", "Android"),
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Mobile/15E148 Instagram 340.0.0.22.109": ("mobile", "Other", "iOS"),
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Mobile/15E148 [LinkedInApp]/9.30": ("mobile", "Other", "iOS"),
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.6; rv:131.0) Gecko/20100101 Firefox/131.0":
            ("desktop", "Firefox", "macOS"),
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 "
        "Safari/537.36 Edg/129.0": ("desktop", "Edge", "Windows"),
        "Mozilla/5.0 (Linux; Android 13; SM-X710) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 "
        "Safari/537.36": ("tablet", "Chrome", "Android"),
    }
    for ua in people:
        assert hit(c, ua=ua).status_code == 204, ua
    got = [(e["device"], e["browser"], e["os"]) for e in reversed(stored(store))]
    assert got == list(people.values())


def test_hit_body_is_capped_at_2_kb(monkeypatch):
    c, store, _ = make(monkeypatch)
    big = body(title="x" * 2100)
    assert hit(c, big).status_code == 413  # declared Content-Length

    def chunks():  # chunked upload: no Content-Length to check first
        for _ in range(5):
            yield b"x" * 600

    r = c.post("/api/hit", content=chunks(), headers={"Origin": SITE, "User-Agent": CHROME,
                                                      "Content-Type": "text/plain"})
    assert r.status_code == 413
    assert stored(store) == []
    fits = body(title="t" * 100)
    fits = fits[:-1] + ', "pad": "' + "p" * (visits.MAX_BODY - len(fits) - 12) + '"}'
    assert len(fits.encode()) <= visits.MAX_BODY
    assert hit(c, fits).status_code == 204 and len(stored(store)) == 1


def test_hit_bad_bodies_are_refused(monkeypatch):
    c, store, _ = make(monkeypatch)
    for payload in ("not json", "[1, 2]", "null", json.dumps({"path": "no-slash"}), json.dumps({"title": "x"}),
                    b"\xff\xfe"):
        r = hit(c, payload)
        assert r.status_code == 400 and r.json()["error"]["code"] == "bad_request", payload
    assert stored(store) == []


def test_hit_text_fields_lose_control_and_bidi_characters(monkeypatch):
    c, store, _ = make(monkeypatch)
    assert hit(c, body(title="Hi\x1b[31m red\u202e evil\n", path="/a\x07b/")).status_code == 204
    [ev] = stored(store)
    assert "\x1b" not in ev["title"] and "\u202e" not in ev["title"] and "\n" not in ev["title"]
    assert ev["path"] == "/a b/"


def test_hit_caps_per_ip_and_global_reset_next_day(monkeypatch):
    clock = Clock()
    c, store, _ = make(monkeypatch, clock=clock, HIT_IP_CAP=2, HIT_DAILY_CAP=3)
    assert [hit(c, ip="198.51.100.1").status_code for _ in range(3)] == [204, 204, 429]
    assert hit(c, ip="198.51.100.2").status_code == 204       # 3rd hit of the day overall
    r = hit(c, ip="198.51.100.3")                             # global cap
    assert r.status_code == 429 and r.json()["error"]["code"] == "hit_cap"
    assert len(stored(store)) == 3
    clock.t += 86400
    assert hit(c, ip="198.51.100.1").status_code == 204
    assert len(stored(store)) == 4


def test_hit_cap_defaults_keep_a_month_of_fake_hits_inside_the_upstash_free_tier(monkeypatch):
    fake = FakeUpstash()
    c, _, app = make(monkeypatch, store=visits.UpstashStore("https://x.upstash.io", "tok", post=fake))
    caps = app.state.hit_caps
    assert (caps.ip_cap, caps.daily_cap) == (100, 3000) == (visits.HIT_IP_CAP, visits.HIT_DAILY_CAP)
    for i in range(10):
        assert hit(c, ip=f"198.51.100.{i}").status_code == 204
    assert sum(len(call["json"]) for call in fake.calls) == 10 * 4 + 1  # 4 commands a hit + one EXPIRE a day
    assert (caps.daily_cap * 4 + 1) * 31 < 500_000  # Upstash free tier: 500K commands a month
    assert caps.daily_cap < visits.MAX_EVENTS  # a day at the cap cannot fill the whole log


def test_hit_keeps_only_the_allowed_query_parameters(monkeypatch):
    c, store, _ = make(monkeypatch)
    qs = ("?site=client.com&mc_eid=abc123&email=a%40b.c&utm_source=upwork&_hsenc=p2ANqtz&li_fat_id=9f&"
          "color=%232a78d6&gclid=x&q=refund+policy&sel=n3&Site=caps&theme=dark&utm_medium=proposal&"
          "utm_campaign=oct&utm_term=t&me=1&hittest=1&api=http%3A%2F%2Flocalhost")
    assert hit(c, body(qs=qs)).status_code == 204
    assert stored(store)[0]["qs"] == ("?site=client.com&utm_source=upwork&color=%232a78d6&q=refund+policy&sel=n3&"
                                      "theme=dark&utm_medium=proposal&utm_campaign=oct")
    for junk in ("?email=a%40b.c&mc_eid=1", "?", "site=x", "?=x&&"):
        assert hit(c, body(qs=junk)).status_code == 204
        assert stored(store)[0]["qs"] == "", junk
    # filtered before the 200-character cut: a long junk parameter cannot push a kept one out
    assert hit(c, body(qs="?fbclid=" + "x" * 900 + "&site=b.example")).status_code == 204
    assert stored(store)[0]["qs"] == "?site=b.example"
    assert visits.clean_qs("?site=" + "s" * 300) == "?site=" + "s" * (visits.MAX_QS - 6)
    assert visits.clean_qs("?utm%5Fsource=x&a=1") == "?utm%5Fsource=x"  # keys compared decoded, kept as sent


def test_hit_caps_keep_no_plain_ip():
    caps = visits.HitCaps(ip_cap=5, daily_cap=10)
    assert caps.take("192.0.2.9")
    assert "192.0.2.9" not in repr(caps.__dict__)


def test_a_broken_store_never_breaks_the_hit(monkeypatch):
    class Boom(visits.MemoryStore):
        def add(self, event):
            raise RuntimeError("store exploded")

    c, _, _ = make(monkeypatch, store=Boom())
    assert hit(c).status_code == 204
    down = visits.UpstashStore("https://x.upstash.io", "tok", post=FakeUpstash(fail="raise"))
    c, _, _ = make(monkeypatch, store=down)
    assert hit(c).status_code == 204


def test_other_posts_still_accept_requests_without_origin(monkeypatch):
    c, _, _ = make(monkeypatch)  # the Origin-required rule is for /api/hit only
    r = c.post("/api/kb", content=json.dumps({"url": "ftp://x"}), headers={"Content-Type": "text/plain"})
    assert r.status_code != 403


# ---------------------------------------------------------------- GET /api/stats

def test_stats_is_hidden_without_a_key_and_refuses_a_wrong_one(monkeypatch):
    c, _, _ = make(monkeypatch, stats_key="")
    assert c.get("/api/stats").status_code == 404
    assert c.get("/api/stats", headers={"X-Stats-Key": ""}).status_code == 404
    c, _, _ = make(monkeypatch, stats_key="s3cret-key")
    assert c.get("/api/stats").status_code == 401
    assert c.get("/api/stats?days=abc", headers={"X-Stats-Key": "wrong"}).status_code == 401
    r = c.get("/api/stats", headers={"X-Stats-Key": "s3cret-key "})
    assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"
    r = c.get("/api/stats?days=abc&limit=-3", headers={"X-Stats-Key": "s3cret-key"})
    assert r.status_code == 200 and r.json()["days"] == 7 and r.json()["events"] == []


def _visit_log(monkeypatch, store):
    clock = Clock()  # 2027-01-15 08:00 UTC
    c, store, _ = make(monkeypatch, store=store, clock=clock)
    clock.t -= 3 * 86400  # 2027-01-12: outside a 2-day window
    hit(c, body(path="/old/", sid="old1"), **{"x-vercel-ip-country": "FR", "x-vercel-ip-city": "Paris"})
    clock.t += 2 * 86400  # 2027-01-14
    hit(c, body(path="/", sid="s1", ref=""), **{"x-vercel-ip-country": "US", "x-vercel-ip-city": "New%20York"})
    clock.t += 60
    hit(c, body(path="/ask/", sid="s1", ref=SITE + "/"), **{"x-vercel-ip-country": "US",
                                                           "x-vercel-ip-city": "New%20York"})
    clock.t += 86400  # 2027-01-15
    hit(c, body(path="/ask/", sid="s2"), **{"x-vercel-ip-country": "MA", "x-vercel-ip-city": "Casablanca"})
    hit(c, body(path="/rag-fix/", sid="s3"))  # no geo headers: country unknown
    return c, store


@pytest.mark.parametrize("kind", ["memory", "upstash"])
def test_stats_totals_and_newest_events(monkeypatch, kind):
    fake = FakeUpstash()
    store = visits.MemoryStore() if kind == "memory" else visits.UpstashStore("https://x.upstash.io", "tok", post=fake)
    c, store = _visit_log(monkeypatch, store)
    r = c.get("/api/stats?days=2&limit=3", headers={"X-Stats-Key": "s3cret-key"})
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    d = r.json()
    assert d["store"] == kind and (d["from"], d["to"], d["days"]) == ("2027-01-14", "2027-01-15", 2)
    assert d["by_day"] == [{"day": "2027-01-14", "views": 2, "sessions": 1},
                           {"day": "2027-01-15", "views": 2, "sessions": 2}]
    assert d["views"] == 4 and d["sessions"] == 3 and d["events_in_period"] == 4 and d["partial"] is False
    assert d["by_country"] == [{"key": "US", "n": 2}, {"key": "??", "n": 1}, {"key": "MA", "n": 1}]
    assert d["by_page"] == [{"key": "/ask/", "n": 2}, {"key": "/", "n": 1}, {"key": "/rag-fix/", "n": 1}]
    assert d["by_city"] == [{"key": "New York, US", "n": 2}, {"key": "(unknown), ??", "n": 1},
                            {"key": "Casablanca, MA", "n": 1}]
    assert d["by_ref"] == [{"key": "www.upwork.com", "n": 2}, {"key": "(direct)", "n": 1},
                           {"key": "slimanechanchoul.com", "n": 1}]
    assert [e["path"] for e in d["events"]] == ["/rag-fix/", "/ask/", "/ask/"]  # newest first, limit 3
    assert IP not in r.text and "Mozilla" not in r.text
    if kind == "upstash":
        assert all(call["url"] == "https://x.upstash.io/pipeline" for call in fake.calls)
        writes = [call["json"] for call in fake.calls if call["json"][0][0] == "LPUSH"]
        assert [cmd[0] for cmd in writes[0]] == ["LPUSH", "LTRIM", "HINCRBY", "HINCRBY", "EXPIRE"]
        assert writes[0][1] == ["LTRIM", "visits:events", "0", "4999"]
        assert writes[0][4] == ["EXPIRE", "visits:day:2027-01-12", str(visits.DAY_TTL_S)]
        assert [len(w) for w in writes] == [5, 5, 4, 5, 4]  # EXPIRE on each day's first write only
        assert fake.calls[0]["timeout"] == visits.WRITE_TIMEOUT


def test_stats_503_when_the_store_cannot_be_read(monkeypatch):
    c, _, _ = make(monkeypatch, store=visits.UpstashStore("https://x.upstash.io", "tok", post=FakeUpstash(fail="500")))
    r = c.get("/api/stats", headers={"X-Stats-Key": "s3cret-key"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "store_unavailable"


def test_stats_says_partial_when_the_log_is_full():
    store = visits.MemoryStore(max_events=3)
    now = 1_800_000_000
    for i in range(5):
        store.add(visits.make_event({"path": f"/p{i}/", "sid": f"s{i}"}, {}, CHROME, now + i))
    d = visits.stats(store, now + 10, days=1, limit=10, chunk=2)
    assert d["partial"] is True and d["events_in_period"] == 3 and d["views"] == 5
    assert [e["path"] for e in d["events"]] == ["/p4/", "/p3/", "/p2/"]


# ---------------------------------------------------------------- stores

def test_upstash_failures_are_swallowed():
    ev = visits.make_event({"path": "/"}, {}, CHROME, 1_800_000_000)
    for fail in ("raise", "500", "garbage"):
        s = visits.UpstashStore("https://x.upstash.io", "tok", post=FakeUpstash(fail=fail))
        assert s.add(ev) is False and s.lrange(0, 9) is None and s.day_hashes(["2027-01-15"]) is None
    wrong = visits.UpstashStore("https://x.upstash.io", "bad", post=FakeUpstash(token="tok"))
    assert wrong.add(ev) is False and wrong.lrange(0, 9) is None
    assert "bad" not in repr(wrong) and "tok" not in repr(visits.UpstashStore("https://x.upstash.io", "tok"))


def test_upstash_ring_buffer_keeps_the_last_events():
    fake = FakeUpstash()
    s = visits.UpstashStore("https://x.upstash.io/", "tok", post=fake, max_events=3)
    for i in range(5):
        assert s.add(visits.make_event({"path": f"/p{i}/"}, {}, CHROME, 1_800_000_000 + i))
    assert [json.loads(x)["path"] for x in s.lrange(0, 99)] == ["/p4/", "/p3/", "/p2/"]
    assert s.day_hashes(["2027-01-15", "2027-01-16"]) == [
        {"c:??": 5, "p:/p0/": 1, "p:/p1/": 1, "p:/p2/": 1, "p:/p3/": 1, "p:/p4/": 1}, {}]
    assert fake.calls[0]["headers"] == {"Authorization": "Bearer tok"}


def test_upstash_sends_expire_once_a_day_and_again_after_a_failed_write():
    fake = FakeUpstash(fail="500")
    s = visits.UpstashStore("https://x.upstash.io", "tok", post=fake)
    t = 1_800_000_000  # 2027-01-15 08:00 UTC

    def add(ts, path="/"):
        ok = s.add(visits.make_event({"path": path}, {}, CHROME, ts))
        return ok, [cmd[0] for cmd in fake.calls[-1]["json"]]

    assert add(t) == (False, ["LPUSH", "LTRIM", "HINCRBY", "HINCRBY", "EXPIRE"])
    fake.fail = None
    assert add(t + 1) == (True, ["LPUSH", "LTRIM", "HINCRBY", "HINCRBY", "EXPIRE"])  # the failed one did not count
    assert add(t + 2, "/a/") == (True, ["LPUSH", "LTRIM", "HINCRBY", "HINCRBY"])
    assert add(t + 86400) == (True, ["LPUSH", "LTRIM", "HINCRBY", "HINCRBY", "EXPIRE"])  # next day
    assert add(t + 3) == (True, ["LPUSH", "LTRIM", "HINCRBY", "HINCRBY"])  # late write of the day before
    assert fake.ttl == {"visits:day:2027-01-15": visits.DAY_TTL_S, "visits:day:2027-01-16": visits.DAY_TTL_S}
    other = visits.UpstashStore("https://x.upstash.io", "tok", post=fake)  # each instance sends it once
    other.add(visits.make_event({"path": "/"}, {}, CHROME, t + 86400 + 5))
    assert fake.calls[-1]["json"][-1] == ["EXPIRE", "visits:day:2027-01-16", str(visits.DAY_TTL_S)]
    for day in range(2, 30):
        add(t + day * 86400)
    assert s._ttl_days == {"visits:day:2027-02-11", "visits:day:2027-02-12", "visits:day:2027-02-13"}  # the last 3 days


def test_memory_ring_buffer_keeps_the_last_events():
    s = visits.MemoryStore(max_events=2)
    for i in range(4):
        s.add(visits.make_event({"path": f"/p{i}/"}, {}, CHROME, 1_800_000_000 + i))
    assert [json.loads(x)["path"] for x in s.lrange(0, 99)] == ["/p3/", "/p2/"]
    assert s.day_hashes(["2027-01-15"])[0]["c:??"] == 4


def test_store_from_env():
    assert isinstance(visits.from_env({}), visits.MemoryStore)
    s = visits.from_env({"KV_REST_API_URL": "https://a.upstash.io", "KV_REST_API_TOKEN": " 'tok' "})
    assert isinstance(s, visits.UpstashStore) and s.url == "https://a.upstash.io" and s._token == "tok"
    s = visits.from_env({"UPSTASH_REDIS_REST_URL": "https://b.upstash.io/", "UPSTASH_REDIS_REST_TOKEN": "t2"})
    assert isinstance(s, visits.UpstashStore) and s.url == "https://b.upstash.io"
    assert isinstance(visits.from_env({"KV_REST_API_URL": "https://a.upstash.io"}), visits.MemoryStore)
    assert isinstance(visits.from_env({"KV_REST_API_URL": "redis://x", "KV_REST_API_TOKEN": "t"}), visits.MemoryStore)


# ---------------------------------------------------------------- scripts/visits.py

def _cli():
    spec = importlib.util.spec_from_file_location("visits_cli", ROOT / "scripts" / "visits.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_reads_the_key_from_the_env_file(tmp_path):
    cli = _cli()
    env = tmp_path / ".env"
    env.write_text("github_token=gh\nopenrouter_key=or\nexport stats_key = 'k-123'\n")
    assert cli.read_key(env, environ={}) == "k-123"
    assert cli.read_key(env, environ={"STATS_KEY": "from-env"}) == "from-env"
    assert cli.read_key(tmp_path / "missing", environ={}) == ""


def test_cli_prints_a_table_and_never_the_key(monkeypatch, tmp_path, capsys):
    cli = _cli()
    monkeypatch.delenv("STATS_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("stats_key=super-secret-value\n")
    seen = {}
    data = {"store": "upstash", "days": 7, "from": "2027-01-09", "to": "2027-01-15", "views": 3, "sessions": 2,
            "by_day": [{"day": "2027-01-15", "views": 3, "sessions": 2}],
            "by_country": [{"key": "US", "n": 2}, {"key": "MA", "n": 1}], "by_page": [{"key": "/ask/", "n": 3}],
            "by_ref": [{"key": "www.upwork.com", "n": 2}], "by_city": [{"key": "New York, US", "n": 2}],
            "events": [{"utc": "2027-01-15T08:00:00Z", "sid": "abcd1234", "country": "US", "city": "New York",
                        "path": "/ask/", "qs": "?site=x.example", "ref": "www.upwork.com", "device": "desktop",
                        "browser": "Chrome", "title": "\x1b[2Jboom"}],
            "events_in_period": 3, "partial": False}

    def fake_fetch(api, key, days, limit):
        seen.update(api=api, key=key, days=days, limit=limit)
        return data

    monkeypatch.setattr(cli, "fetch", fake_fetch)
    assert cli.main(["--env", str(env), "--days", "7", "--rows", "5"]) == 0
    out = capsys.readouterr().out
    assert seen == {"api": cli.DEFAULT_API, "key": "super-secret-value", "days": 7, "limit": 5}
    assert "super-secret-value" not in out and "\x1b" not in out
    assert "01-15 08:00:00" in out and "New York" in out and "/ask/?site=x.example" in out and "www.upwork.com" in out
    assert "By country" in out and "By page" in out and "3 page views" in out


def test_cli_errors_say_what_to_fix_without_the_key(monkeypatch, tmp_path, capsys):
    cli = _cli()
    monkeypatch.delenv("STATS_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("stats_key=super-secret-value\n")

    class FakeHTTPError(cli.urllib.error.HTTPError):
        def __init__(self, code):
            super().__init__("https://api.example/api/stats", code, "x", {}, None)

    def refuse(req, timeout=None):
        assert req.unredirected_hdrs.get("X-stats-key") == "super-secret-value"  # not re-sent on a redirect
        raise FakeHTTPError(401)

    monkeypatch.setattr(cli.urllib.request, "urlopen", refuse)
    assert cli.main(["--env", str(env)]) == 1
    err = capsys.readouterr().err
    assert "refused the key" in err and "super-secret-value" not in err
    assert cli.main(["--env", str(env), "--api", "http://api.example"]) == 1  # no key over plain http
    assert "https://" in capsys.readouterr().err
    assert cli.main(["--env", str(tmp_path / "none")]) == 2
    assert "stats_key" in capsys.readouterr().err
