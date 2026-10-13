"""PDF reader tests: tiny PDFs built in memory, no files on disk."""
import io
import sys
from pathlib import Path

import pypdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docs import is_scanned_pdf, read_pdfs  # noqa: E402

TEXT = "Our return policy allows returns within thirty days of delivery for a full refund to the original payment method"


def build_pdf(content: bytes, resources: bytes, extra: list[bytes] = ()) -> bytes:
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources " + resources + b" >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        *extra,
    ]
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return out


def text_pdf(text=TEXT) -> bytes:
    return build_pdf(f"BT /F1 12 Tf 72 712 Td ({text}) Tj ET".encode(), b"<< /Font << /F1 5 0 R >> >>",
                     [b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"])


def scan_pdf() -> bytes:
    image = b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 8 /Length 1 >>\nstream\n\xff\nendstream"
    return build_pdf(b"q 612 0 0 792 0 0 cm /Im1 Do Q", b"<< /XObject << /Im1 5 0 R >> >>", [image])


def named(data: bytes, name: str) -> io.BytesIO:
    f = io.BytesIO(data)
    f.name = name
    return f


def test_scanned_pdf_gets_its_own_message():
    pages, problems = read_pdfs([named(scan_pdf(), "scan.pdf")])
    assert pages == []
    assert problems == ["scan.pdf: looks scanned (images only). Client versions add OCR."]


def test_is_scanned_pdf():
    assert is_scanned_pdf(pypdf.PdfReader(io.BytesIO(scan_pdf())))
    assert not is_scanned_pdf(pypdf.PdfReader(io.BytesIO(text_pdf())))
    assert not is_scanned_pdf(pypdf.PdfReader(io.BytesIO(build_pdf(b"", b"<< >>"))))


def test_empty_pdf_without_images_keeps_old_message():
    _, problems = read_pdfs([named(build_pdf(b"", b"<< >>"), "blank.pdf")])
    assert problems == ["blank.pdf: no text found (scanned PDFs need OCR, which client versions include)"]


def test_page_extraction_crash_is_a_problem_not_a_traceback(monkeypatch):
    def boom(self, *a, **kw):
        raise RuntimeError("bad font")
    monkeypatch.setattr(pypdf.PageObject, "extract_text", boom)
    pages, problems = read_pdfs([named(text_pdf(), "odd.pdf")])
    assert pages == []
    assert problems[0] == "odd.pdf: skipped 1 unreadable page."


def test_password_pdf_and_empty_password_pdf():
    def encrypted(user: str) -> bytes:
        w = pypdf.PdfWriter(clone_from=pypdf.PdfReader(io.BytesIO(text_pdf())))
        w.encrypt(user_password=user, owner_password="owner", algorithm="RC4-128")
        out = io.BytesIO()
        w.write(out)
        return out.getvalue()

    pages, problems = read_pdfs([named(encrypted("secret"), "locked.pdf"), named(encrypted(""), "limits.pdf")])
    assert problems == ["locked.pdf: password-protected"]
    assert [p.title for p in pages] == ["limits.pdf, page 1"]


@pytest.mark.parametrize("data", [b"not a pdf", b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 9 0 R >> endobj\ntrailer << /Root 1 0 R >>\n%%EOF"])
def test_broken_files_never_raise(data):
    pages, problems = read_pdfs([named(data, "broken.pdf")])
    assert pages == [] and problems and problems[0].startswith("broken.pdf: ")
