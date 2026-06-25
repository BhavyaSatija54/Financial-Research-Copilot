"""
Financial Research Copilot — Streamlit UI

Multi-page application:
  🔍 Research     — conversational RAG query interface
  📥 Ingest       — ingest SEC filings, FRED macro data, or custom files
  📊 Dashboard    — index statistics and latency charts
  📚 Documents    — browse indexed documents
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests
import streamlit as st

# ── Page config must be first ─────────────────────────────────────────
st.set_page_config(
    page_title="Financial Research Copilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY  = os.getenv("API_KEY", "change-me")
HEADERS  = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d2e 0%, #141624 100%);
        border-right: 1px solid #2d3748;
    }

    /* Cards */
    .metric-card {
        background: #1e2130;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card h2 { color: #63b3ed; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #a0aec0; font-size: 0.85rem; margin: 0; }

    /* Chat bubbles */
    .user-bubble {
        background: #2b6cb0;
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0 0.5rem 20%;
        box-shadow: 0 2px 8px rgba(43,108,176,0.3);
    }
    .assistant-bubble {
        background: #1e2130;
        color: #e2e8f0;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 18px 4px;
        border: 1px solid #2d3748;
        margin: 0.5rem 20% 0.5rem 0;
    }

    /* Citation chips */
    .citation-chip {
        display: inline-block;
        background: #2d3748;
        border: 1px solid #4a5568;
        border-radius: 6px;
        padding: 0.25rem 0.6rem;
        font-size: 0.78rem;
        color: #90cdf4;
        margin: 2px;
    }

    /* Status badges */
    .badge-ok      { background:#276749; color:#9ae6b4; padding:2px 8px; border-radius:4px; font-size:0.8rem; }
    .badge-error   { background:#742a2a; color:#fc8181; padding:2px 8px; border-radius:4px; font-size:0.8rem; }
    .badge-warn    { background:#744210; color:#fbd38d; padding:2px 8px; border-radius:4px; font-size:0.8rem; }

    /* Input override */
    .stTextInput > div > div > input { background: #1e2130; color: #e2e8f0; border-color: #4a5568; }
    .stTextArea > div > div > textarea { background: #1e2130; color: #e2e8f0; border-color: #4a5568; }

    /* Buttons */
    .stButton > button { background: #2b6cb0; color: white; border: none; border-radius: 8px; font-weight: 600; }
    .stButton > button:hover { background: #2c5282; }

    /* Section headers */
    .section-header { color: #63b3ed; font-size: 1.1rem; font-weight: 700;
                      border-bottom: 1px solid #2d3748; padding-bottom: 0.4rem; margin-bottom: 1rem; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────

def api_get(path: str, params: dict | None = None) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", headers=HEADERS, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to API. Make sure the server is running: `make run-api`")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, payload: dict, timeout: int = 120) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", headers=HEADERS,
                          data=json.dumps(payload), timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to API. Make sure the server is running: `make run-api`")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def stream_query(question: str, filters: dict) -> tuple[str, list, float]:
    """Call /api/v1/query/stream and collect SSE tokens."""
    payload = {"query": question, "filters": filters or None}
    answer_parts: list[str] = []
    citations: list = []
    latency_ms: float = 0.0

    try:
        with requests.post(
            f"{API_BASE}/api/v1/query/stream",
            headers=HEADERS,
            data=json.dumps(payload),
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line and line.startswith(b"data: "):
                    try:
                        event = json.loads(line[6:])
                        if event.get("type") == "token":
                            answer_parts.append(event["token"])
                        elif event.get("type") == "done":
                            citations = event.get("citations", [])
                            latency_ms = event.get("latency_ms", 0)
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        return f"Stream error: {e}", [], 0.0

    return "".join(answer_parts), citations, latency_ms


# ── Sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 Financial Research Copilot")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🔍 Research", "📥 Ingest", "📊 Dashboard", "📚 Documents"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    health = api_get("/health")
    if health:
        s = health.get("index_stats", {})
        st.markdown(f"**Index Status** <span class='badge-ok'>●  Online</span>", unsafe_allow_html=True)
        st.caption(f"Vectors: {s.get('faiss_vectors', 0):,}")
        st.caption(f"Docs: {s.get('sqlite_documents', 0):,}")
        st.caption(f"Tickers: {s.get('sqlite_tickers', 0):,}")
    else:
        st.markdown("<span class='badge-error'>●  Offline</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**API**: `" + API_BASE + "`")
    st.markdown("[📖 API Docs](" + API_BASE + "/docs)", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE: RESEARCH
# ══════════════════════════════════════════════════════════════════════
if page == "🔍 Research":
    st.title("🔍 Financial Research Assistant")
    st.caption("Ask questions about SEC filings, earnings transcripts, and macroeconomic data.")

    # Filters in expander
    with st.expander("🔧 Search Filters", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            ticker_filter  = st.text_input("Ticker", placeholder="AAPL").upper().strip()
        with col2:
            type_filter = st.selectbox("Filing Type", ["Any", "10-K", "10-Q", "8-K", "MACRO"])
        with col3:
            date_from = st.date_input("Date From", value=None)
            date_to   = st.date_input("Date To",   value=None)

    filters: dict = {}
    if ticker_filter: filters["ticker"] = ticker_filter
    if type_filter != "Any": filters["filing_type"] = type_filter
    if date_from: filters["date_from"] = str(date_from)
    if date_to:   filters["date_to"]   = str(date_to)

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Suggested questions
    if not st.session_state.messages:
        st.markdown("### 💡 Try asking…")
        suggestions = [
            "What are the key risk factors in Apple's 2024 10-K?",
            "Summarise revenue and margin trends for MSFT over the last 3 years",
            "What is the current Federal Reserve interest rate stance?",
            "Compare AAPL and GOOGL gross margins in recent filings",
            "What does the yield curve inversion signal for the economy?",
            "What guidance did NVDA management provide for next quarter?",
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            if cols[i % 2].button(s, key=f"sug_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": s})
                st.session_state._run_query = s
                st.rerun()

    # Render history
    for msg in st.session_state.messages:
        cls = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
        st.markdown(f'<div class="{cls}">{msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("citations"):
            with st.expander(f"📚 {len(msg['citations'])} Sources"):
                for c in msg["citations"]:
                    company = c.get("company", "Unknown")
                    ftype   = c.get("filing_type", "")
                    period  = c.get("period", "")
                    score   = c.get("relevance_score", 0)
                    url     = c.get("source_url", "")
                    link    = f'<a href="{url}" target="_blank">{company}</a>' if url else company
                    st.markdown(
                        f'<span class="citation-chip">[{c["citation_number"]}] '
                        f'{link} | {ftype} | {period} | score: {score:.3f}</span>',
                        unsafe_allow_html=True,
                    )

    # Input
    col_input, col_clear = st.columns([5, 1])
    with col_input:
        question = st.chat_input("Ask a financial research question…")
    with col_clear:
        if st.button("Clear", use_container_width=True) and st.session_state.messages:
            st.session_state.messages = []
            st.rerun()

    # Run pending suggestion or typed question
    run_q = getattr(st.session_state, "_run_query", None) or question
    if run_q:
        if hasattr(st.session_state, "_run_query"):
            del st.session_state._run_query

        if not st.session_state.messages or st.session_state.messages[-1]["content"] != run_q:
            st.session_state.messages.append({"role": "user", "content": run_q})

        with st.spinner("🔍 Researching…"):
            t0 = time.perf_counter()
            answer, citations, latency_ms = stream_query(run_q, filters)
            if not latency_ms:
                latency_ms = (time.perf_counter() - t0) * 1000

        if answer and not answer.startswith("Stream error"):
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "citations": citations,
            })
            st.rerun()
        elif answer:
            st.error(answer)


# ══════════════════════════════════════════════════════════════════════
# PAGE: INGEST
# ══════════════════════════════════════════════════════════════════════
elif page == "📥 Ingest":
    st.title("📥 Document Ingestion")
    st.caption("Index SEC filings, macroeconomic data, or custom documents.")

    tab_sec, tab_macro, tab_file, tab_text = st.tabs(
        ["🏛️ SEC EDGAR", "📈 FRED Macro", "📁 Upload File", "✏️ Raw Text"]
    )

    # ── SEC tab ──────────────────────────────────────────────────────
    with tab_sec:
        st.markdown("#### Fetch SEC EDGAR Filings")
        st.caption("Downloads directly from EDGAR using the public API. No API key required.")
        col1, col2, col3 = st.columns(3)
        with col1:
            sec_ticker = st.text_input("Ticker Symbol", value="AAPL", placeholder="AAPL")
        with col2:
            sec_type   = st.selectbox("Filing Type", ["10-K", "10-Q", "8-K"])
        with col3:
            sec_limit  = st.slider("Number of filings", 1, 10, 3)

        if st.button("🚀 Fetch & Ingest", key="btn_sec"):
            if not sec_ticker:
                st.warning("Enter a ticker symbol")
            else:
                with st.spinner(f"Fetching {sec_ticker} {sec_type} filings from SEC EDGAR…"):
                    result = api_post("/api/v1/ingest/sec", {
                        "ticker": sec_ticker.upper(),
                        "filing_type": sec_type,
                        "limit": sec_limit,
                    }, timeout=180)
                if result:
                    if result["status"] in ("success", "partial"):
                        st.success(f"✅ Ingested **{result['documents_processed']}** filings — "
                                   f"{result['chunks_added']:,} chunks / {result['tokens_indexed']:,} tokens")
                    if result.get("errors"):
                        for e in result["errors"]:
                            st.warning(f"⚠️ {e}")

    # ── FRED tab ─────────────────────────────────────────────────────
    with tab_macro:
        st.markdown("#### Fetch FRED Macroeconomic Data")
        st.caption("Uses the St. Louis Fed public API. Add `FRED_API_KEY` to `.env` for full access.")

        fred_series_options = {
            "GDP — US Gross Domestic Product": "GDP",
            "CPIAUCSL — Consumer Price Index": "CPIAUCSL",
            "PCEPILFE — Core PCE (Fed's preferred)": "PCEPILFE",
            "FEDFUNDS — Federal Funds Rate": "FEDFUNDS",
            "UNRATE — Unemployment Rate": "UNRATE",
            "GS10 — 10-Year Treasury Yield": "GS10",
            "GS2 — 2-Year Treasury Yield": "GS2",
            "T10Y2Y — Yield Curve Spread": "T10Y2Y",
            "INDPRO — Industrial Production": "INDPRO",
            "UMCSENT — Consumer Sentiment": "UMCSENT",
            "HOUST — Housing Starts": "HOUST",
            "M2SL — M2 Money Supply": "M2SL",
        }

        selected = st.multiselect(
            "Select series (leave empty for all defaults)",
            options=list(fred_series_options.keys()),
            default=["GDP — US Gross Domestic Product",
                     "CPIAUCSL — Consumer Price Index",
                     "FEDFUNDS — Federal Funds Rate",
                     "UNRATE — Unemployment Rate",
                     "GS10 — 10-Year Treasury Yield",
                     "T10Y2Y — Yield Curve Spread"],
        )
        obs_start = st.text_input("Observation start date", value="2018-01-01")

        if st.button("🚀 Fetch & Ingest FRED Data", key="btn_macro"):
            series_ids = [fred_series_options[s] for s in selected] if selected else None
            with st.spinner("Fetching macroeconomic data from FRED…"):
                result = api_post("/api/v1/ingest/macro", {
                    "series_ids": series_ids,
                    "observation_start": obs_start,
                }, timeout=120)
            if result:
                st.success(f"✅ Ingested **{result['documents_processed']}** macro series — "
                           f"{result['chunks_added']:,} chunks")
                if result.get("errors"):
                    for e in result["errors"]:
                        st.warning(f"⚠️ {e}")

    # ── File upload tab ───────────────────────────────────────────────
    with tab_file:
        st.markdown("#### Upload Document")
        st.caption("Supports plain text (.txt). PDF text extraction coming soon.")
        uploaded = st.file_uploader("Choose a file", type=["txt"])
        if uploaded:
            col1, col2 = st.columns(2)
            with col1: st.info(f"📄 {uploaded.name} ({len(uploaded.getvalue()):,} bytes)")
            if st.button("🚀 Ingest File", key="btn_file"):
                with st.spinner(f"Ingesting {uploaded.name}…"):
                    r = requests.post(
                        f"{API_BASE}/api/v1/ingest/file",
                        headers={"X-API-Key": API_KEY},
                        files={"file": (uploaded.name, uploaded.getvalue(), "text/plain")},
                        timeout=120,
                    )
                if r.status_code == 200:
                    d = r.json()
                    st.success(f"✅ {d['chunks_added']:,} chunks indexed")
                else:
                    st.error(f"Error {r.status_code}: {r.text}")

    # ── Raw text tab ──────────────────────────────────────────────────
    with tab_text:
        st.markdown("#### Paste Raw Text")
        raw_doc_id = st.text_input("Document ID", placeholder="AAPL_custom_note_2024")
        raw_text   = st.text_area("Document text", height=200,
                                   placeholder="Paste your document content here…")
        raw_ticker      = st.text_input("Ticker (optional)", placeholder="AAPL")
        raw_filing_type = st.text_input("Filing type (optional)", placeholder="10-K")

        if st.button("🚀 Ingest Text", key="btn_text"):
            if not raw_doc_id or not raw_text or len(raw_text) < 50:
                st.warning("Provide a doc ID and at least 50 characters of text.")
            else:
                meta: dict = {}
                if raw_ticker:      meta["ticker"]      = raw_ticker.upper()
                if raw_filing_type: meta["filing_type"] = raw_filing_type.upper()
                with st.spinner("Ingesting…"):
                    result = api_post("/api/v1/ingest/text", {
                        "doc_id": raw_doc_id, "text": raw_text, "metadata": meta,
                    })
                if result:
                    st.success(f"✅ {result['chunks_added']:,} chunks indexed")


# ══════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    import plotly.graph_objects as go

    st.title("📊 Index Dashboard")

    health = api_get("/health")
    if not health:
        st.stop()

    stats = health.get("index_stats", {})
    latency = health.get("latency_stats", {})

    # ── KPI cards ────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        (c1, stats.get("faiss_vectors", 0), "FAISS Vectors"),
        (c2, stats.get("sqlite_documents", 0), "Documents"),
        (c3, stats.get("sqlite_tickers", 0), "Tickers"),
        (c4, stats.get("sqlite_chunks", 0), "SQLite Chunks"),
    ]
    for col, val, label in kpis:
        col.markdown(
            f'<div class="metric-card"><h2>{val:,}</h2><p>{label}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # Filing type breakdown
    with col_left:
        st.markdown('<p class="section-header">📋 Chunks by Filing Type</p>', unsafe_allow_html=True)
        by_type = stats.get("by_filing_type", {})
        if by_type:
            fig = go.Figure(go.Pie(
                labels=list(by_type.keys()),
                values=list(by_type.values()),
                hole=0.45,
                marker_colors=["#63b3ed", "#68d391", "#f6ad55", "#fc8181", "#b794f4"],
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", showlegend=True,
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=20, b=20, l=20, r=20), height=300,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No documents indexed yet.")

    # Latency breakdown
    with col_right:
        st.markdown('<p class="section-header">⚡ Latency Percentiles (ms)</p>', unsafe_allow_html=True)
        if latency:
            ops = list(latency.keys())
            p50 = [latency[o].get("p50_ms", 0) for o in ops]
            p95 = [latency[o].get("p95_ms", 0) for o in ops]
            p99 = [latency[o].get("p99_ms", 0) for o in ops]
            fig2 = go.Figure(data=[
                go.Bar(name="p50", x=ops, y=p50, marker_color="#68d391"),
                go.Bar(name="p95", x=ops, y=p95, marker_color="#f6ad55"),
                go.Bar(name="p99", x=ops, y=p99, marker_color="#fc8181"),
            ])
            fig2.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", xaxis=dict(gridcolor="#2d3748"),
                yaxis=dict(gridcolor="#2d3748"), margin=dict(t=20, b=20),
                legend=dict(bgcolor="rgba(0,0,0,0)"), height=300,
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No latency data yet. Run some queries first.")

    # System info
    st.markdown("---")
    st.markdown('<p class="section-header">⚙️ System Configuration</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("LLM Model",        stats.get("llm_model", "—"))
    c2.metric("Embedding Model",  stats.get("embedding_model", "—"))
    c3.metric("Cache",
              "✅ Redis" if health.get("cache_available") else "⚪ Disabled")

    env   = health.get("environment", "—")
    ver   = health.get("version", "—")
    badge = "badge-ok" if health.get("status") == "ok" else "badge-error"
    st.markdown(
        f'Environment: <span class="{badge}">{env}</span> &nbsp; Version: `{ver}`',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════
# PAGE: DOCUMENTS
# ══════════════════════════════════════════════════════════════════════
elif page == "📚 Documents":
    st.title("📚 Indexed Documents")
    st.caption("Browse all documents currently in the index.")

    col1, col2, col3 = st.columns(3)
    with col1: filter_ticker  = st.text_input("Filter by ticker", placeholder="AAPL")
    with col2: filter_type    = st.selectbox("Filter by type", ["All", "10-K", "10-Q", "8-K", "MACRO", "upload"])
    with col3: doc_limit      = st.slider("Max results", 10, 200, 50)

    params: dict = {"limit": doc_limit}
    if filter_ticker: params["ticker"]      = filter_ticker.upper()
    if filter_type != "All": params["filing_type"] = filter_type

    docs_data = api_get("/api/v1/ingest/documents", params=params)

    if docs_data:
        docs = docs_data.get("documents", [])
        st.caption(f"Showing {len(docs)} of {docs_data.get('total', 0)} documents")

        if docs:
            import pandas as pd
            df = pd.DataFrame(docs)
            display_cols = ["doc_id", "ticker", "company", "filing_type", "period",
                            "filed_date", "chunk_count"]
            present = [c for c in display_cols if c in df.columns]
            st.dataframe(
                df[present].rename(columns={
                    "doc_id": "Document ID", "ticker": "Ticker",
                    "company": "Company", "filing_type": "Type",
                    "period": "Period", "filed_date": "Filed Date",
                    "chunk_count": "Chunks",
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No documents match the current filters. Try ingesting some data first.")
            if st.button("Go to Ingest →"):
                st.session_state.page = "📥 Ingest"
                st.rerun()
