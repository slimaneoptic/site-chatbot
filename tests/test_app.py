"""The live Streamlit app wired to the new engine: no network (fake crawl, fake model)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import llm  # noqa: E402
import scraper  # noqa: E402
import streamlit as st  # noqa: E402
from scraper import CrawlError, CrawlReport, Page  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

SHOP = [
    Page("https://shop.example/help/shipping", "Shipping and delivery",
         "We deliver everywhere in Morocco. Orders to Casablanca and Rabat arrive in 1 to 2 working days. "
         "Shipping costs 35 MAD per order and is free for orders over 500 MAD."),
    Page("https://shop.example/help/payment", "Payment",
         "We accept Visa and Mastercard bank cards, PayPal, and cash on delivery in Morocco."),
]


class FakeModel:
    def __init__(self, text="Orders arrive in 1 to 2 working days [1].\nFOLLOWUPS: How much is shipping? | Can I pay by card?"):
        self.text, self.calls = text, []

    def __call__(self, messages, timeout=90, *, max_tokens=4000, deadline_s=None, on_usage=None, session_id=None):
        self.calls.append(messages)
        yield from (self.text[:20], self.text[20:])
        if on_usage:
            on_usage(llm.Usage(prompt_tokens=100, completion_tokens=20, cost_usd=0.0001, model="fake", calls=1))


@pytest.fixture
def app(monkeypatch):
    st.cache_data.clear()
    st.cache_resource.clear()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-a-key")
    model, crawled = FakeModel(), []

    def fake_read_site(url, max_pages=15, on_page=None, **kw):
        crawled.append(url)
        return CrawlReport(url, list(SHOP), 2, "done", False, [])

    monkeypatch.setattr(scraper, "read_site", fake_read_site)
    monkeypatch.setattr(llm, "stream_text", model)
    monkeypatch.setattr(llm, "translate_query", lambda text, lang, **kw: text)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20)
    at.model, at.crawled = model, crawled
    return at


def load(at, url="shop.example/help/"):
    at.run()
    at.text_input[0].set_value(url)
    at.button[0].click()  # "Read website"
    return at.run()


def texts(at) -> str:
    return "\n".join(str(getattr(e, "value", "")) for e in list(at.markdown) + list(at.caption) + list(at.info) + list(at.warning) + list(at.error))


def test_starts_without_a_site(app):
    app.run()
    assert not app.exception and "Read website" in texts(app)


def test_url_without_scheme_is_read(app):
    load(app)
    assert not app.exception
    assert app.crawled == ["https://shop.example/help/"]
    assert "shop.example" in texts(app) and app.status[0].label.startswith("Read 2 pages")


def test_crawl_error_shows_friendly_message(app, monkeypatch):
    def fail(url, max_pages=15, on_page=None, **kw):
        raise CrawlError("unreachable")

    monkeypatch.setattr(scraper, "read_site", fail)
    load(app, "nowhere.example")
    assert not app.exception
    assert app.status[0].label == scraper.MESSAGES["unreachable"]


def test_unexpected_crawl_failure_has_no_traceback(app, monkeypatch):
    def boom(url, max_pages=15, on_page=None, **kw):
        raise RuntimeError("internal detail")

    monkeypatch.setattr(scraper, "read_site", boom)
    load(app, "nowhere.example")
    assert not app.exception and "internal detail" not in app.status[0].label


def test_bad_url_is_explained(app):
    load(app, "not a url")
    assert not app.exception and app.error and "spaces" in app.error[0].value
    assert app.crawled == []


def test_js_only_warning_is_shown(app, monkeypatch):
    monkeypatch.setattr(scraper, "read_site", lambda url, max_pages=15, on_page=None, **kw:
                        CrawlReport(url, [], 1, "done", True, [scraper.MESSAGES["js_only"]]))
    load(app, "spa.example")
    assert not app.exception and any("JavaScript" in w.value for w in app.warning)


def test_off_topic_question_gets_no_match_card_without_model_call(app):
    load(app)
    app.chat_input[0].set_value("Who won the 2022 World Cup?").run()
    assert not app.exception
    assert "couldn't find that on this site" in texts(app)
    assert app.model.calls == []


def test_answer_with_sources_followups_and_cache(app):
    load(app)
    app.chat_input[0].set_value("How long does delivery to Casablanca take?").run()
    assert not app.exception and len(app.model.calls) == 1
    out = texts(app)
    assert "1 to 2 working days" in out and "**Sources**" in out and "FOLLOWUPS" not in out
    labels = [b.label for b in app.button]
    assert "How much is shipping?" in labels
    next(b for b in app.button if b.label == "How much is shipping?").click().run()  # follow-up button asks it
    assert not app.exception and len(app.model.calls) == 2
    assert app.session_state.chat[-2]["content"] == "How much is shipping?"


def test_session_cap_gives_passages_and_never_says_reload(app, monkeypatch):
    load(app)
    for i in range(13):
        app.chat_input[0].set_value("Delivery to Casablanca and Rabat?").run()  # history is not empty after the first: no cache
    assert not app.exception
    assert len(app.model.calls) == 12  # session cap
    out = texts(app)
    assert "used the AI answers for this visit" in out and "reload" not in out.lower()


def test_same_question_from_another_visitor_comes_from_the_cache(app):
    load(app)
    app.chat_input[0].set_value("Can I pay with PayPal?").run()
    assert len(app.model.calls) == 1
    other = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20)  # a second session, same process
    load(other)
    other.chat_input[0].set_value("can I pay with paypal").run()
    assert not other.exception and len(app.model.calls) == 1  # no second model call
    assert other.session_state.chat[-1]["status"] == "cached"


@pytest.mark.parametrize("forwarded, hops, peer, want", [
    ("6.6.6.6, 203.0.113.7", None, "10.0.0.1", "203.0.113.7"),  # the visitor's fake first entry is ignored
    ("6.6.6.6, 203.0.113.7, 10.1.1.1", "2", "10.0.0.1", "203.0.113.7"),
    ("", None, "198.51.100.4", "198.51.100.4"),  # no proxy header: Streamlit's own TCP peer
])
def test_visitor_ip_uses_trusted_proxy_hop(app, monkeypatch, forwarded, hops, peer, want):
    import budget
    from streamlit.runtime.context import ContextProxy, StreamlitHeaders
    monkeypatch.setattr(ContextProxy, "headers", property(lambda self: StreamlitHeaders(
        [("X-Forwarded-For", forwarded)] if forwarded else [])))
    monkeypatch.setattr(ContextProxy, "ip_address", property(lambda self: peer))
    if hops:
        monkeypatch.setenv("TRUSTED_PROXY_HOPS", hops)
    else:
        monkeypatch.delenv("TRUSTED_PROXY_HOPS", raising=False)
    seen, real = [], budget.Budget.check_crawl
    monkeypatch.setattr(budget.Budget, "check_crawl", lambda self, ip: seen.append(ip) or real(self, ip))
    load(app)
    assert not app.exception and seen == [want]
