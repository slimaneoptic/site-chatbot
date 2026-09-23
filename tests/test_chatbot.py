import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import llm  # noqa: E402
from answer import AnswerError, extractive_answer, stream_answer  # noqa: E402
from scraper import BlockedURL, Page, _links, _scope, check_public_url  # noqa: E402
from search import Index, chunk  # noqa: E402

PAGES = [
    Page("https://example.com/shipping", "Shipping", "We ship to Morocco, Saudi Arabia and the UAE.\n\nDelivery takes 3 to 5 working days. " * 3),
    Page("https://example.com/returns", "Returns", "You can return any item within 30 days for a full refund.\n\nContact support to start a return. " * 3),
]


@pytest.mark.parametrize("url", ["http://127.0.0.1/", "http://10.1.2.3/", "http://169.254.169.254/", "ftp://example.com/", "http://example.com:8080/"])
def test_internal_and_odd_urls_are_blocked(url):
    with pytest.raises(BlockedURL):
        check_public_url(url)


def test_scope_and_link_extraction():
    assert _scope("https://docs.site.com/guide/intro.html") == ("docs.site.com", "/guide/")
    html = '<a href="/guide/a.html#x">a</a><a href="b.pdf">b</a><a href="https://other.com/">c</a>'
    assert _links(html, "https://docs.site.com/guide/") == ["https://docs.site.com/guide/a.html", "https://other.com/"]


def test_chunking_and_search_find_the_right_page():
    assert all(len(c.text) <= 1100 for p in PAGES for c in chunk(p))
    top = Index(PAGES).top("how many days to return an item for a refund", 2)
    assert top[0].url.endswith("/returns")
    assert Index(PAGES).top("zzzz unrelated", 2) == []


def test_extractive_answer_links_sources():
    text = extractive_answer(Index(PAGES).top("delivery days", 3))
    assert "https://example.com/shipping" in text


def test_stream_answer_sends_excerpts_and_turns_busy_into_answer_error():
    seen = {}

    def fake_stream(messages):
        seen["messages"] = messages
        yield from ["Returns are accepted ", "within 30 days [1]."]

    out = "".join(stream_answer("example.com", "Can I return?", Index(PAGES).top("return refund", 2), [], stream=fake_stream))
    assert out == "Returns are accepted within 30 days [1]."
    assert seen["messages"][0]["role"] == "system"
    assert "[1] Returns (https://example.com/returns)" in seen["messages"][-1]["content"]

    def busy(messages):
        raise llm.LLMError("All AI models are busy right now.")
        yield  # pragma: no cover

    with pytest.raises(AnswerError):
        "".join(stream_answer("example.com", "x", [], [], stream=busy))


def test_table_extraction_cleans_rows_and_sends_exact_fields():
    from extract import extract_rows, parse_fields

    fields = parse_fields("Quote, author, Author")
    assert fields == [("quote", "Quote"), ("author", "author")]
    seen = {}

    def fake_complete(messages, schema, name):
        seen["schema"] = schema
        return {"rows": [{"quote": " Be yourself. ", "author": "Oscar Wilde"}, {"quote": "", "author": ""}]}, "fake"

    rows = extract_rows("https://q.example/", "some page text", fields, complete=fake_complete)
    assert rows == [{"Quote": "Be yourself.", "author": "Oscar Wilde"}]
    item = seen["schema"]["properties"]["rows"]["items"]
    assert item["required"] == ["quote", "author"] and item["additionalProperties"] is False


def _tiny_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 712 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return out


def test_pdfs_become_citable_pages():
    import io

    from docs import read_pdfs

    text = "Our return policy allows returns within thirty days of delivery for a full refund to the original payment method"
    f = io.BytesIO(_tiny_pdf(text))
    f.name = "policy.pdf"
    bad = io.BytesIO(b"not a pdf")
    bad.name = "broken.pdf"
    pages, problems = read_pdfs([f, bad])
    assert [p.title for p in pages] == ["policy.pdf, page 1"]
    assert "thirty days" in pages[0].text
    assert problems == ["broken.pdf: not a readable PDF"]
    assert "policy.pdf, page 1" in extractive_answer(Index(pages).top("return policy refund", 1))
