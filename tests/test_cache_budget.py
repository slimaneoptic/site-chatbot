import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from budget import Budget  # noqa: E402
from cache import AnswerCache, normalize_question, source_id  # noqa: E402
from llm import Usage  # noqa: E402
from scraper import Page  # noqa: E402


def test_source_id_is_order_independent_and_text_sensitive():
    a, b = Page("https://x/a", "A", "one"), Page("https://x/b", "B", "two")
    assert source_id([a, b]) == source_id([b, a]) and len(source_id([a])) == 16
    assert source_id([a]) != source_id([Page("https://x/a", "A", "one!")])


def test_normalize_question():
    assert normalize_question("  How   much is SHIPPING?! ") == "how much is shipping"
    assert normalize_question("كم ثمن الشحن؟") == "كم ثمن الشحن"


def test_cache_lru_and_ttl():
    now = [1000.0]
    c = AnswerCache(max_entries=2, ttl_s=60, clock=lambda: now[0])
    c.put("s", "Q1", "r1"); c.put("s", "Q2", "r2")
    assert c.get("s", "q1?") == "r1"  # q1 is now the most recent
    c.put("s", "Q3", "r3")
    assert c.get("s", "q2") is None and c.get("s", "q3") == "r3" and c.get("other", "q3") is None
    now[0] += 61
    assert c.get("s", "q1") is None


def test_budget_caps_and_daily_reset():
    now = [1_760_000_000.0]
    b = Budget(session_cap=2, ip_cap=3, daily_usd=0.01, ip_crawl_cap=1, clock=lambda: now[0])
    one = Usage(cost_usd=0.001, calls=1)
    assert b.check("s1", "ip").allowed
    b.record("s1", "ip", one); b.record("s1", "ip", one)
    d = b.check("s1", "ip")
    assert (d.allowed, d.reason) == (False, "session_cap") and d.message and "reload" not in d.message.lower()
    b.record("s2", "ip", one)
    assert b.check("s3", "ip").reason == "ip_cap" and b.check("s3", "ip2").allowed
    b.record("s9", "ip9", Usage(cost_usd=0.02, calls=0))  # cost counts, question does not
    assert b.check("s4", "ip4").reason == "global_budget"
    assert b.snapshot()["questions"] == 3
    b.record_crawl("ip")
    assert b.check_crawl("ip").reason == "ip_crawl_cap" and b.check_crawl("ip2").allowed
    now[0] += 86_400
    assert b.check("s1", "ip").allowed and b.check_crawl("ip").allowed
    assert b.snapshot()["spent_usd"] == 0 and b.snapshot()["crawls"] == 0
