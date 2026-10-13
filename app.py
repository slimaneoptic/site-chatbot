"""Website chatbot + scraper demo: crawl a site, download clean data, chat with its content."""
from __future__ import annotations

import json
import os
import uuid
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

try:  # Streamlit Cloud secrets -> environment
    for name in ("OPENROUTER_API_KEY", "OPENROUTER_MODELS", "DEMO_DAILY_USD", "TRUSTED_PROXY_HOPS"):
        if name in st.secrets:
            os.environ.setdefault(name, str(st.secrets[name]))  # TOML ints (TRUSTED_PROXY_HOPS = 2) too
except Exception:
    pass

import llm  # noqa: E402
import scraper  # noqa: E402
from answer import NOT_FOUND_TEXT, ai_available, ask, extractive_answer  # noqa: E402
from budget import Budget, pick_client_ip  # noqa: E402
from cache import AnswerCache, source_id  # noqa: E402
from docs import read_pdfs  # noqa: E402
from extract import extract_rows, parse_fields  # noqa: E402
from scraper import CrawlError, Page, normalize_url  # noqa: E402
from search import Index  # noqa: E402

# ---- edit these two lines -------------------------------------------------
AUTHOR = "S'limane"
HIRE_URL = "https://www.upwork.com/freelancers/~01d6ae0f1a06a43ac0"
# ---------------------------------------------------------------------------

MAX_QUESTIONS = 12  # AI answers per visitor session; after that the best passages are shown
MAX_IP_QUESTIONS = 30  # AI answers per network per day
MAX_EXTRACT_PAGES = 20  # per visitor session
DAILY_USD = float(os.environ.get("DEMO_DAILY_USD") or 1.0)  # whole demo, per UTC day
QUOTES = "https://quotes.toscrape.com/"
EXAMPLES = {"Python tutorial": "https://docs.python.org/3/tutorial/", "FastAPI tutorial": "https://fastapi.tiangolo.com/tutorial/"}
AI_BUSY = "The AI is busy right now, so here are the most relevant passages instead."
CRAWL_FAILED = "Something went wrong while reading that site. Try again or try another address."

st.set_page_config(page_title="Website chatbot & scraper", page_icon="💬", layout="wide")
st.markdown(
    """<style>
    .block-container {padding-top: 2rem; max-width: 1100px;}
    .footer {margin-top:2.5rem; padding-top:1rem; border-top:1px solid #D5DBE1; color:#4B5866; font-size:.92rem;}
    .site-line {color:#4B5866; font-size:.95rem; margin:.2rem 0 1rem;}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def shared() -> tuple[AnswerCache, Budget]:
    """One answer cache and one budget for the whole process (all visitors)."""
    return AnswerCache(), Budget(session_cap=MAX_QUESTIONS, ip_cap=MAX_IP_QUESTIONS, daily_usd=DAILY_USD)


CACHE, BUDGET = shared()
SID = st.session_state.setdefault("sid", uuid.uuid4().hex)


def visitor_ip() -> str:
    """Client address for the daily caps, with the same rule as api/main.py: the leftmost X-Forwarded-For entry
    is whatever the visitor sent, so take the TRUSTED_PROXY_HOPS-th entry from the right (default 1 = the one
    our hosting proxy wrote); with no header, Streamlit's own TCP peer (st.context.ip_address)."""
    try:
        hops = max(0, int(os.environ.get("TRUSTED_PROXY_HOPS") or 1))
    except ValueError:
        hops = 1
    try:
        headers = st.context.headers
        forwarded = headers.get_all("X-Forwarded-For") if hasattr(headers, "get_all") else [headers.get("X-Forwarded-For") or ""]
        return pick_client_ip(forwarded, st.context.ip_address, hops)
    except Exception:  # noqa: BLE001 - bare mode / tests
        return "unknown"


IP = visitor_ip()


@st.cache_data(ttl=3600, show_spinner=False, max_entries=20)
def scrape(url: str, max_pages: int) -> dict:
    """Read a site once per hour per address. CrawlError is raised (and never cached)."""
    st.session_state["_crawled"] = True
    status = st.session_state.get("_status")

    def progress(n, u):
        if status:
            status.update(label=f"Read {n} page{'s' * (n != 1)}… {u[:80]}")
    report = scraper.read_site(url, max_pages, on_page=progress)
    return {"start_url": report.start_url, "pages": [vars(p) for p in report.pages],
            "warnings": list(report.warnings), "js_only": report.js_only}


def start_scrape(url: str | None = None):
    """Button callback. Reads the box and slider here, at click time: args bound on the previous run
    would miss an address typed just before the click."""
    if url:
        st.session_state.url_box = url  # an example button also fills the box
    st.session_state.pending = (st.session_state.get("url_box") or "", st.session_state.get("max_pages", 12))
    st.session_state.open_tab = None


def start_extract_example():
    st.session_state.url_box = QUOTES
    st.session_state.pending = (QUOTES, 6)
    st.session_state.fields = "quote, author"
    st.session_state.open_tab = "Extract a table"


def ask_followup(question: str):
    st.session_state.queued = question


def source_ref(title: str, url: str) -> str:
    return f"[{title}]({url})" if url.startswith("http") else title


def load_site(raw: str, n: int) -> None:
    """Read the address the visitor gave; friendly messages for every failure, never a traceback."""
    try:
        target = normalize_url(raw)
    except CrawlError as err:
        st.error(err.message)
        return
    is_example = target in EXAMPLES.values() or target == QUOTES
    if not is_example:
        decision = BUDGET.check_crawl(IP)
        if not decision.allowed:
            st.warning(decision.message)
            return
    report = None
    st.session_state["_crawled"] = False
    with st.status(f"Reading {target}…", expanded=False) as status:
        st.session_state._status = status
        try:
            report = scrape(target, n)
            got = len(report["pages"])
            status.update(label=f"Read {got} page{'s' * (got != 1)} from {urlparse(report['start_url']).netloc}",
                          state="complete" if got else "error")
        except CrawlError as err:
            status.update(label=err.message, state="error")
        except Exception:  # noqa: BLE001 - a visitor never sees a stack trace
            status.update(label=CRAWL_FAILED, state="error")
        finally:
            st.session_state._status = None
    if report is None:
        return
    if st.session_state.get("_crawled") and not is_example and report["pages"]:
        BUDGET.record_crawl(IP)
    for w in report["warnings"]:
        st.warning(w)
    if not report["pages"]:
        if not report["warnings"]:
            st.warning(scraper.MESSAGES["no_text"])
        return  # keep the site that was loaded before
    st.session_state.update(pages=report["pages"], site=report["start_url"], site_label=urlparse(report["start_url"]).netloc,
                            site_notes=report["warnings"], chat=[], extracted=None, open_tab=None)


def answer_question(index: Index, site: str, source: str, question: str) -> dict:
    """Search, then answer with at most one model call. Renders into the current chat message and
    returns the assistant message to keep in the chat."""
    chat = st.session_state.chat
    prev_q = next((m["plain"] for m in reversed(chat) if m["role"] == "user"), None)
    ai_on = ai_available()
    allowed = BUDGET.check(SID, IP).allowed
    tr_usage: list[llm.Usage] = []
    translate = (lambda text, lang: llm.translate_query(text, lang, on_usage=tr_usage.append)) if ai_on and allowed else None
    found = index.search(question, prev_question=prev_q, translate=translate)
    for u in tr_usage:  # the translation costs money but is not an answer: cost only, no question counted
        BUDGET.record(SID, IP, llm.Usage(cost_usd=u.cost_usd, model=u.model))

    if not found.matched:
        topics = list(dict.fromkeys(p.title for p in index.passages if p.title))[:3]
        hint = "Nothing on the pages read matches that question. Try other words"
        hint += f", or ask about {', '.join(topics)}." if topics else "."
        st.markdown(f"**{NOT_FOUND_TEXT}**")
        st.caption(hint)
        return {"role": "assistant", "content": f"**{NOT_FOUND_TEXT}**", "plain": NOT_FOUND_TEXT, "note": hint,
                "history": False, "status": "no_match"}

    if not ai_on:
        text = extractive_answer(found.passages)
        st.markdown(text)
        return {"role": "assistant", "content": text, "plain": text, "history": False, "status": "extractive"}

    history = [{"role": m["role"], "content": m["plain"]} for m in chat if m.get("history", True)]
    note_slot = st.empty()
    run = ask(site, question, found, history, source=source, session=SID, ip=IP,
              cache=CACHE, budget=BUDGET, stream=llm.stream_text)
    with st.spinner("Thinking…"):
        st.write_stream(iter(run))
    res = run.result
    note = AI_BUSY if res.note == "AI busy" else res.note
    if note:
        note_slot.info(note)
    reply = res.text
    if res.status in ("answered", "cached"):
        lines = [f"[{i}] {source_ref(res.passages[i - 1].title, res.passages[i - 1].url)}" for i in res.cited]
        if lines:
            sources = "**Sources**  \n" + "  \n".join(lines)
            st.markdown(sources)
            reply = f"{res.text}\n\n{sources}"
    good = res.status in ("answered", "cached")
    return {"role": "assistant", "content": reply, "plain": res.text, "note": note, "history": good,
            "followups": res.followups if good else [], "status": res.status}


st.title("Website chatbot & scraper")
st.markdown(
    "Paste a website or upload PDFs. It reads them, answers questions **only from that content** with sources, "
    "pulls the fields you name into a table, and gives you the clean text as a spreadsheet."
)

c1, c2, c3 = st.columns([4, 1.4, 1.2], vertical_alignment="bottom")
st.session_state.setdefault("url_box", EXAMPLES["Python tutorial"])
c1.text_input("Website", key="url_box", placeholder="yourcompany.com/help/")
c2.slider("Pages to read", 3, 30, 12, key="max_pages")
c3.button("Read website", type="primary", width="stretch", on_click=start_scrape)
e1, e2, _ = st.columns([1.2, 1.2, 4])
for col, (label, link) in zip((e1, e2), EXAMPLES.items()):
    col.button(f"Try: {label}", on_click=start_scrape, args=(link,), width="stretch")
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
                                    site_notes=problems, chat=[], extracted=None, open_tab=None)

if "pending" in st.session_state:
    load_site(*st.session_state.pop("pending"))

pages = [Page(**p) for p in st.session_state.get("pages", [])]
if not pages:
    st.info("Pick an example or paste a website address, then press **Read website**.")
else:
    source = source_id(pages)
    if st.session_state.get("index_source") != source:
        st.session_state.index, st.session_state.index_source = Index(pages), source
    index = st.session_state.index
    site = st.session_state.get("site_label") or urlparse(st.session_state.get("site", "")).netloc
    st.markdown(f'<div class="site-line">{len(pages)} pages · {sum(p.words for p in pages):,} words · {len(index.passages)} passages from <b>{site}</b></div>', unsafe_allow_html=True)
    for note in st.session_state.get("site_notes") or []:
        st.caption(note)
    chat_tab, extract_tab, data_tab = st.tabs(["Chat", "Extract a table", f"Scraped data ({len(pages)} pages)"], default=st.session_state.get("open_tab") or "Chat")

    with chat_tab:
        if not ai_available():
            st.caption("AI answers are off in this deployment, so it shows the most relevant passages. With an API key it writes a short answer with numbered sources.")
        st.session_state.setdefault("chat", [])
        conversation = st.container()
        typed = st.chat_input(f"Ask something about {site}")
        question = typed or st.session_state.pop("queued", None)
        with conversation:
            chat = st.session_state.chat
            for msg in chat:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("note"):
                        st.caption(msg["note"])
            if question:
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    reply = answer_question(index, site, source, question)
                user_msg = {"role": "user", "content": question, "plain": question, "history": reply["history"]}
                st.session_state.chat = chat = chat + [user_msg, reply]
            last = chat[-1] if chat else None
            if last and last["role"] == "assistant" and last.get("followups"):
                st.caption("Ask next")
                cols = st.columns(len(last["followups"]))
                for i, (col, q) in enumerate(zip(cols, last["followups"])):
                    col.button(q, key=f"fu-{len(chat)}-{i}", on_click=ask_followup, args=(q,), width="stretch")

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
                    left = MAX_EXTRACT_PAGES - used
                    st.warning(f"This demo extracts from up to {MAX_EXTRACT_PAGES} pages per visit"
                               + (f"; {left} page{'s' * (left != 1)} left, so pick fewer pages." if left > 0 else ", and they are used up. The chat and the data download still work."))
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
- **Scraper:** crawls pages under the address you give, respects robots.txt, waits between requests, extracts the main text and drops menus, footers and ads. It tells you when a site needs JavaScript to show its text.
- **PDFs:** reads text-based PDFs page by page and cites "file.pdf, page N"; scanned (image-only) PDFs are flagged.
- **Search:** splits pages into passages and ranks them against the question (BM25+), uses your previous question for follow-ups, and translates a question asked in another language than the site's.
- **Extract a table:** the AI reads each page and returns rows with exactly the fields you asked for, validated against a strict schema, with the source page on every row.
- **Answers:** an AI model writes a short answer from the best passages only, cites them, and suggests what to ask next. A question the pages don't cover gets an honest "not found" with no AI call; repeated questions come from a cache. If the AI is busy, you get the best passages instead.
- **Client versions:** PDFs and help-center exports, scheduled re-indexing, an embeddable chat widget, lead capture and WhatsApp or email handoff, analytics on unanswered questions.
"""
    )

st.markdown(
    f'<div class="footer">Built by {AUTHOR}. I build scrapers, data pipelines and AI assistants on your own content. '
    f'<a href="{HIRE_URL}" target="_blank">Work with me</a></div>',
    unsafe_allow_html=True,
)
