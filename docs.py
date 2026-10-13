"""Read uploaded PDFs into the same Page objects the website crawler produces."""
from __future__ import annotations

import logging

from pypdf import PdfReader

from scraper import Page

MAX_PAGES = 300
MIN_WORDS = 10          # pages with fewer words are blank or near-empty
SCAN_CHECK_PAGES = 20   # how many pages is_scanned_pdf() looks at

log = logging.getLogger(__name__)


def _page_text(pg) -> str | None:
    """Text of one page, or None when pypdf can't read that page."""
    try:
        return (pg.extract_text() or "").strip()
    except Exception:  # noqa: BLE001 - one broken page must not stop the file
        log.warning("PDF page text extraction failed", exc_info=True)
        return None


def _has_image(obj, depth: int = 0) -> bool:
    """True when a page (or a form inside it) draws at least one image."""
    try:
        res = obj.get("/Resources")
        res = res.get_object() if res is not None else None
        xobjs = res.get("/XObject") if res is not None else None
        xobjs = xobjs.get_object() if xobjs is not None else None
        for name in xobjs or {}:
            x = xobjs[name].get_object()
            if x.get("/Subtype") == "/Image":
                return True
            if x.get("/Subtype") == "/Form" and depth < 3 and _has_image(x, depth + 1):
                return True
    except Exception:  # noqa: BLE001
        return False
    return False


def is_scanned_pdf(reader) -> bool:
    """True when the first pages have no real text but do contain images (a scan that needs OCR)."""
    has_image = False
    try:
        for i, pg in enumerate(reader.pages):
            if i >= SCAN_CHECK_PAGES:
                break
            if len((_page_text(pg) or "").split()) >= MIN_WORDS:
                return False
            has_image = has_image or _has_image(pg)
    except Exception:  # noqa: BLE001
        return False
    return has_image


def _unlock(reader) -> bool:
    """Open PDFs that are encrypted with an empty user password (print/copy limits only)."""
    try:
        return bool(reader.decrypt(""))
    except Exception:  # noqa: BLE001 - e.g. AES without the optional crypto package
        return False


def read_pdfs(files) -> tuple[list[Page], list[str]]:
    """Return (pages, problems). Each PDF page becomes one Page titled 'file.pdf, page N'."""
    pages, problems = [], []
    for f in files:
        name = getattr(f, "name", None) or "file.pdf"
        try:
            reader = PdfReader(f)
            if reader.is_encrypted and not _unlock(reader):
                problems.append(f"{name}: password-protected")
                continue
        except Exception:  # noqa: BLE001 - pypdf raises many types on broken files
            problems.append(f"{name}: not a readable PDF")
            continue
        found = unreadable = 0
        has_image = False
        try:
            for i, pg in enumerate(reader.pages, start=1):
                if len(pages) >= MAX_PAGES:
                    problems.append(f"Stopped at {MAX_PAGES} pages (demo limit).")
                    return pages, problems
                text = _page_text(pg)
                if text is None:
                    unreadable += 1
                elif len(text.split()) >= MIN_WORDS:  # skip blank or near-empty pages
                    pages.append(Page(f"{name}#page={i}", f"{name}, page {i}", text))
                    found += 1
                if not found and not has_image and i <= SCAN_CHECK_PAGES:
                    has_image = _has_image(pg)
        except Exception:  # noqa: BLE001
            log.warning("PDF page list broken", exc_info=True)
            problems.append(f"{name}: part of the file is damaged; read what was possible.")
        if unreadable:
            problems.append(f"{name}: skipped {unreadable} unreadable page{'s' * (unreadable != 1)}.")
        if not found:
            if has_image:
                problems.append(f"{name}: looks scanned (images only). Client versions add OCR.")
            else:
                problems.append(f"{name}: no text found (scanned PDFs need OCR, which client versions include)")
    return pages, problems
