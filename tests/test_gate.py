"""Tapemark phone-test reports: POST /api/gate, GET /api/gate-results, the store methods (Upstash faked)."""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import visits  # noqa: E402
from test_visits import CHROME, IP, IPHONE, SITE, FakeUpstash, make  # noqa: E402

TEXT = "Room engine phone test (gate 1.3.0)\nSelf-test: PASS (3 of 3 rooms match Python).\n"


def report(**kw):
    b = {"text": TEXT, "run": "r-1a2b3c4d", "stage": "final", "build": "gate 1.3.0, git abc1234",
         "model": "iPhone 16 Pro Max", "query": "?path=single&stop=first"}
    b.update(kw)
    return json.dumps(b)


def post(c, payload=None, origin=SITE, ua=IPHONE, ip=IP):
    h = {"Content-Type": "text/plain;charset=UTF-8", "User-Agent": ua, "X-Forwarded-For": ip}
    if origin is not None:
        h["Origin"] = origin
    return c.post("/api/gate", content=payload if payload is not None else report(), headers=h)


def reports(store) -> list[dict]:
    return [json.loads(x) for x in store.lrange(0, 10_000, key=visits.GATE_KEY)]


def test_gate_stores_the_report_without_the_ip(monkeypatch):
    c, store, _ = make(monkeypatch)
    r = post(c)
    assert r.status_code == 204
    [rep] = reports(store)
    assert rep["text"] == TEXT and rep["stage"] == "final" and rep["run"] == "r-1a2b3c4d"
    assert rep["model"] == "iPhone 16 Pro Max" and rep["query"] == "?path=single&stop=first"
    assert rep["ua"] == IPHONE and rep["utc"].endswith("Z")
    assert IP not in json.dumps(rep)
    assert store.lrange(0, 99) == []  # page-view log untouched


def test_gate_needs_the_site_origin(monkeypatch):
    c, store, _ = make(monkeypatch)
    assert post(c, origin=None).status_code == 403
    assert post(c, origin="https://evil.example").status_code in (400, 403)
    assert reports(store) == []


def test_gate_refuses_bad_bodies_and_big_ones(monkeypatch):
    c, store, _ = make(monkeypatch)
    assert post(c, payload="not json").status_code == 400
    assert post(c, payload=json.dumps({"text": ""})).status_code == 400
    assert post(c, payload=json.dumps(["x"])).status_code == 400
    assert post(c, payload=report(text="x" * (visits.MAX_GATE_BODY + 10))).status_code == 413
    assert reports(store) == []


def test_gate_cleans_labels_and_unknown_stage(monkeypatch):
    c, store, _ = make(monkeypatch)
    assert post(c, payload=report(stage="weird", run="bad run id!", model="a\x00b" * 50)).status_code == 204
    [rep] = reports(store)
    assert rep["stage"] == "final" and rep["run"] == "" and len(rep["model"]) <= 60 and "\x00" not in rep["model"]


def test_gate_drops_bots_and_caps_per_ip(monkeypatch):
    c, store, _ = make(monkeypatch, HIT_IP_CAP=2)
    assert post(c, ua="Googlebot/2.1 (+http://www.google.com/bot.html)").status_code == 204
    assert reports(store) == []
    assert post(c, ua=CHROME).status_code == 204
    assert post(c, ua=CHROME).status_code == 204
    assert post(c, ua=CHROME).status_code == 429
    assert len(reports(store)) == 2


def test_gate_results_needs_the_key_and_returns_newest_first(monkeypatch):
    c, store, _ = make(monkeypatch)
    post(c, payload=report(run="r-first-0001"))
    post(c, payload=report(run="r-second-02", stage="before-3d"))
    assert c.get("/api/gate-results").status_code == 401
    assert c.get("/api/gate-results", headers={"x-stats-key": "wrong"}).status_code == 401
    r = c.get("/api/gate-results?limit=10", headers={"x-stats-key": "s3cret-key"})
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    assert [x["run"] for x in r.json()["results"]] == ["r-second-02", "r-first-0001"]


def test_gate_results_hidden_without_a_stats_key(monkeypatch):
    c, _, _ = make(monkeypatch, stats_key="")
    assert c.get("/api/gate-results", headers={"x-stats-key": "x"}).status_code == 404


def test_upstash_gate_list_is_capped_and_separate():
    fake = FakeUpstash()
    s = visits.UpstashStore("https://x.upstash.io", "tok", post=fake)
    for i in range(3):
        assert s.add_gate(visits.make_gate({"text": f"t{i}", "run": f"run-{i:06d}"}, IPHONE, 1_800_000_000 + i))
    assert [r["run"] for r in visits.gate_reports(s, 10)] == ["run-000002", "run-000001", "run-000000"]
    assert ["LTRIM", visits.GATE_KEY, "0", str(visits.MAX_GATE - 1)] in fake.calls[0]["json"]
    assert s.lrange(0, 99) == []


def test_a_broken_store_never_breaks_the_gate_post(monkeypatch):
    s = visits.UpstashStore("https://x.upstash.io", "tok", post=FakeUpstash(fail="raise"))
    c, _, _ = make(monkeypatch, store=s)
    assert post(c).status_code == 204
    assert visits.gate_reports(s) is None
