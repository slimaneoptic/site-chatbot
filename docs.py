"""Read uploaded PDFs into the same Page objects the website crawler produces."""
from __future__ import annotations

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from scraper import Page

MAX_PAGES = 300


def read_pdfs(files) -> tuple[list[Page], list[str]]:
    """Return (pages, problems). Each PDF page becomes one Page titled 'file.pdf, page N'."""
    pages, problems = [], []
    for f in files:
        try:
            reader = PdfReader(f)
        except (PdfReadError, ValueError, OSError):
            problems.append(f"{f.name}: not a readable PDF")
            continue
        if reader.is_encrypted:
            problems.append(f"{f.name}: password-protected")
            continue
        found = 0
        for i, pg in enumerate(reader.pages, start=1):
            if len(pages) >= MAX_PAGES:
                problems.append(f"Stopped at {MAX_PAGES} pages (demo limit).")
                return pages, problems
            text = (pg.extract_text() or "").strip()
            if len(text.split()) >= 10:  # skip blank or near-empty pages
                pages.append(Page(f"{f.name}#page={i}", f"{f.name}, page {i}", text))
                found += 1
        if not found:
            problems.append(f"{f.name}: no text found (scanned PDFs need OCR, which client versions include)")
    return pages, problems
