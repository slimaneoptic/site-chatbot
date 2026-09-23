"""Website chatbot + scraper demo: crawl a site, download clean data, chat with its content."""
from __future__ import annotations

import json
import os
import re
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

try:  # Streamlit Cloud secrets -> environment
    for name in ("OPENROUTER_API_KEY", "OPENROUTER_MODELS"):
        if name in st.secrets:
            os.environ.setdefault(name, st.secrets[name])
except Exception:
    pass

import llm  # noqa: E402
from answer import AnswerError, ai_available, extractive_answer, stream_answer  # noqa: E402
from docs import read_pdfs  # noqa: E402
from extract import extract_rows, parse_fields  # noqa: E402
from scraper import BlockedURL, Page, crawl  # noqa: E402
from search import Index  # noqa: E402

# ---- edit these two lines -------------------------------------------------
AUTHOR = "Slimane"
HIRE_URL = "https://www.upwork.com/freelancers/~01d6ae0f1a06a43ac0"
# ---------------------------------------------------------------------------

MAX_QUESTIONS = 12  # per visitor session, keeps a public demo's API bill small
MAX_EXTRACT_PAGES = 20  # per visitor session
QUOTES = "https://quotes.toscrape.com/"
EXAMPLES = {"Python tutorial": "https://docs.python.org/3/tutorial/", "FastAPI tutorial": "https://fastapi.tiangolo.com/tutorial/"}

st.set_page_config(page_title="Website chatbot & scraper", page_icon="💬", layout="wide")
st.markdown(
    """<style>
    .block-container {padding-top: 2rem; max-width: 1100px;}
    .footer {margin-top:2.5rem; padding-top:1rem; border-top:1px solid #D5DBE1; color:#4B5866; font-size:.92rem;}
    .site-line {color:#4B5866; font-size:.95rem; margin:.2rem 0 1rem;}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=20)
def scrape(url: str, max_pages: int) -> list[dict]:
    status = st.session_state.get("_status")
    def progress(n, u):
        if status:
            status.update(label=f"Read {n} page{'s' * (n != 1)}… {u[:80]}")
    return [vars(p) for p in crawl(url, max_pages, on_page=progress)]


def start_scrape(url: str, max_pages: int):
    st.session_state.pending = (url, max_pages)
    st.session_state.open_tab = None


def start_extract_example():
    st.session_state.pending = (QUOTES, 6)
    st.session_state.fields = "quote, author"
    st.session_state.open_tab = "Extract a table"


def source_ref(title: str, url: str) -> str:
    return f"[{title}]({url})" if url.startswith("http") else title


st.title("Website chatbot & scraper")
st.markdown(
    "Paste a website or upload PDFs. It reads them, answers questions **only from that content** with sources, "
    "pulls the fields you name into a table, and gives you the clean text as a spreadsheet."
)

c1, c2, c3 = st.columns([4, 1.4, 1.2], vertical_alignment="bottom")
url = c1.text_input("Website", value=st.session_state.get("url", EXAMPLES["Python tutorial"]), placeholder="https://yourcompany.com/help/")
max_pages = c2.slider("Pages to read", 3, 30, 12)
c3.button("Read website", type="primary", width="stretch", on_click=start_scrape, args=(url, max_pages))
e1, e2, _ = st.columns([1.2, 1.2, 4])
for col, (label, link) in zip((e1, e2), EXAMPLES.items()):
    col.button(f"Try: {label}", on_click=start_scrape, args=(link, max_pages), width="stretch")
st.caption("Reads only pages inside the address you give (same site, same folder), follows robots.txt and waits between requests.")

with st.expander("Or chat with your PDFs instead"):
    pdfs = st.file_uploader("PDF files (text-based, up to 300 pages in total)", type=["pdf"], accept_multiple_files=True)
    if st.button("Read PDFs", disabled=not pdfs):
        with st.spinner("Reading PDFs…"):
            got, problems = read_pdfs(pdfs)
        for p in problems:
            st.warning(p)
        if got:
            st.session_state.update(pages=[vars(p) for p in got], site="", site_label=f"{len(pdfs)} PDF file{'s' * (len(pdfs) != 1)}",
                                    chat=[], asked=0, extracted=None, open_tab=None)

if "pending" in st.session_state:
    target, n = st.session_state.pop("pending")
    st.session_state.url = target
    with st.status(f"Reading {target}…", expanded=False) as status:
        st.session_state._status = status
        try:
            pages = scrape(target, n)
            status.update(label=f"Read {len(pages)} pages from {urlparse(target).netloc}", state="complete")
        except BlockedURL as err:
            pages = None
            status.update(label=str(err), state="error")
        finally:
            st.session_state._status = None
    if pages is not None:
        st.session_state.pages = pages
        st.session_state.site = target
        st.session_state.site_label = urlparse(target).netloc
        st.session_state.chat = []
        st.session_state.asked = 0
        st.session_state.extracted = None
        if not pages:
            st.warning("No readable pages found there. The site may block crawlers, need JavaScript to show text, or keep its content elsewhere. Try a more specific address such as a help or docs section.")

pages = [Page(**p) for p in st.session_state.get("pages", [])]
if not pages:
    st.info("Pick an example or paste a website address, then press **Read website**.")
else:
    index = Index(pages)
    site = st.session_state.get("site_label") or urlparse(st.session_state.site).netloc
    st.markdown(f'<div class="site-line">{len(pages)} pages · {sum(p.words for p in pages):,} words · {len(index.passages)} passages from <b>{site}</b></div>', unsafe_allow_html=True)
    chat_tab, extract_tab, data_tab = st.tabs(["Chat", "Extract a table", f"Scraped data ({len(pages)} pages)"], default=st.session_state.get("open_tab") or "Chat")

    with chat_tab:
        if not ai_available():
            st.caption("AI answers are off in this deployment, so it shows the most relevant passages. With an API key it writes a short answer with numbered sources.")
        conversation = st.container()
        question = st.chat_input(f"Ask something about {site}")
        with conversation:
            for msg in st.session_state.chat:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
        if question:
            with conversation:
                with st.chat_message("user"):
                    st.markdown(question)
                found = index.top(question, 6)
                with st.chat_message("assistant"):
                    if not ai_available():
                        reply = extractive_answer(found)
                        st.markdown(reply)
                    elif st.session_state.asked >= MAX_QUESTIONS:
                        reply = f"This demo answers up to {MAX_QUESTIONS} questions per visit. Reload the page to start over."
                        st.markdown(reply)
                    else:
                        st.session_state.asked += 1
                        history = [{"role": m["role"], "content": m["plain"]} for m in st.session_state.chat]
                        try:
                            with st.spinner("Thinking…"):
                                text = st.write_stream(stream_answer(site, question, found, history))
                        except AnswerError:
                            text = "The AI is busy right now, so here are the most relevant passages instead.\n\n" + extractive_answer(found)
                            st.markdown(text)
                            found = []  # sources are already listed in the passages
                        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(found)})
                        sources = "  \n".join(f"[{i}] {source_ref(found[i - 1].title, found[i - 1].url)}" for i in cited)
                        reply = f"{text}\n\n**Sources**  \n{sources}" if sources else text
                        if sources:
                            st.markdown(f"**Sources**  \n{sources}")
            st.session_state.chat += [
                {"role": "user", "content": question, "plain": question},
                {"role": "assistant", "content": reply, "plain": reply.split("\n\n**Sources**")[0]},
            ]

    with extract_tab:
        st.markdown("Name the fields you want, e.g. *product, price, in stock* or *company, phone, city*. The AI reads each page and fills one row per item.")
        if not ai_available():
            st.info("Table extraction needs the AI model, which isn't configured in this deployment.")
        else:
            f1, f2, f3 = st.columns([3, 1.2, 1.3], vertical_alignment="bottom")
            fields_text = f1.text_input("Fields to extract (comma-separated)", key="fields", placeholder="quote, author")
            n_pages = f2.slider("Pages", 1, min(10, len(pages)), min(5, len(pages))) if len(pages) > 1 else 1
            go = f3.button("Extract table", type="primary", width="stretch")
            st.button("Try it: quotes.toscrape.com → quote, author", on_click=start_extract_example)
            used = st.session_state.get("extract_used", 0)
            if go:
                fields = parse_fields(fields_text or "")
                if not fields:
                    st.warning("Type at least one field name.")
                elif used + n_pages > MAX_EXTRACT_PAGES:
                    st.warning(f"This demo extracts from up to {MAX_EXTRACT_PAGES} pages per visit. Reload the page to start over.")
                else:
                    rows, seen = [], set()
                    with st.status("Extracting…", expanded=False) as status:
                        for i, p in enumerate(pages[:n_pages], start=1):
                            status.update(label=f"Page {i} of {n_pages}: {p.title[:60]}")
                            try:
                                for r in extract_rows(p.url, p.text, fields):
                                    sig = tuple(r.values())
                                    if sig not in seen:
                                        seen.add(sig)
                                        rows.append({**r, "source": p.url})
                            except llm.LLMError:
                                status.update(label=f"The AI is busy right now. Stopped after {i - 1} page(s).", state="error")
                                break
                        else:
                            status.update(label=f"{len(rows)} rows from {n_pages} page(s)", state="complete")
                    st.session_state.extract_used = used + n_pages
                    st.session_state.extracted = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[lab for _, lab in fields] + ["source"])
            table = st.session_state.get("extracted")
            if table is not None:
                if table.empty:
                    st.warning("No matching items found on those pages. Try other field names or more pages.")
                else:
                    st.dataframe(table, hide_index=True, width="stretch", height=360)
                    g1, g2, _ = st.columns([1, 1, 3])
                    g1.download_button("Download CSV", table.to_csv(index=False).encode(), "extracted.csv", "text/csv", width="stretch")
                    g2.download_button("Download JSON", json.dumps(table.to_dict("records"), ensure_ascii=False, indent=1).encode(), "extracted.json", "application/json", width="stretch")

    with data_tab:
        df = pd.DataFrame([{"url": p.url, "title": p.title, "words": p.words, "text": p.text} for p in pages])
        st.dataframe(df.assign(text=df.text.str.slice(0, 160) + "…"), hide_index=True, width="stretch", height=380)
        d1, d2, _ = st.columns([1, 1, 3])
        d1.download_button("Download CSV", df.to_csv(index=False).encode(), f"{site}_pages.csv", "text/csv", width="stretch")
        d2.download_button("Download JSON", json.dumps(df.to_dict("records"), ensure_ascii=False, indent=1).encode(), f"{site}_pages.json", "application/json", width="stretch")

with st.expander("How this works"):
    st.markdown(
        """
- **Scraper:** crawls pages under the address you give, respects robots.txt, waits between requests, extracts the main text and drops menus, footers and ads.
- **PDFs:** reads text-based PDFs page by page and cites "file.pdf, page N".
- **Search:** splits pages into passages and ranks them against the question (BM25+).
- **Extract a table:** the AI reads each page and returns rows with exactly the fields you asked for, validated against a strict schema, with the source page on every row.
- **Answers:** an AI model writes a short answer from the best passages only, cites them, and says so when the site doesn't cover the question. If the AI is busy, you get the best passages instead.
- **Client versions:** PDFs and help-center exports, scheduled re-indexing, an embeddable chat widget, lead capture and WhatsApp or email handoff, analytics on unanswered questions.
"""
    )

st.markdown(
    f'<div class="footer">Built by {AUTHOR}. I build scrapers, data pipelines and AI assistants on your own content. '
    f'<a href="{HIRE_URL}" target="_blank">Work with me</a></div>',
    unsafe_allow_html=True,
)
