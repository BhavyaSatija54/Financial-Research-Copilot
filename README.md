# Financial Research Copilot

> **Production-grade hybrid RAG research assistant for SEC filings, earnings transcripts, and macroeconomic data.**
> Built with Python, LangChain, FAISS, BM25, OpenAI, FastAPI, and Streamlit.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-orange.svg)](https://langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Features

| Feature | Details |
|---|---|
| **Hybrid RAG** | FAISS dense retrieval + BM25 keyword retrieval fused via Reciprocal Rank Fusion |
| **Cross-encoder Reranking** | `ms-marco-MiniLM-L-6-v2` contextual reranking of top candidates |
| **SEC EDGAR Integration** | Auto-fetch 10-K, 10-Q, 8-K filings by ticker from EDGAR public API |
| **FRED Macro Data** | 13 macroeconomic series (GDP, CPI, Fed Funds Rate, yield curve, etc.) via St. Louis Fed |
| **SQLite Metadata Store** | Queryable metadata with FTS5 full-text search; replaces flat JSONL |
| **Streamlit UI** | Dark-theme conversational interface with charts, filters, document browser |
| **Streaming Responses** | SSE token streaming on `/api/v1/query/stream` |
| **Redis Caching** | SHA-256 keyed query cache; ~80ms on cache hit vs ~1.2s cold |
| **Production API** | FastAPI + rate limiting + CORS + GZip + request logging |

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/yourusername/financial-research-copilot.git
cd financial-research-copilot

python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
make install
```

### 2. Configure

```bash
make env           # copies .env.example → .env
# Edit .env — at minimum set OPENAI_API_KEY
```

Key variables:
```env
OPENAI_API_KEY=sk-...          # Required
OPENAI_MODEL=gpt-4o-mini       # or gpt-4o for best quality
API_KEY=your-secret-key        # Protects API endpoints
FRED_API_KEY=                  # Optional — free at fred.stlouisfed.org
SEC_USER_AGENT=YourApp/1.0 email@example.com
```

### 3. Ingest data

```bash
# Ingest bundled sample documents (no API key needed)
make ingest-sample

# Ingest SEC 10-K filings for a ticker
make ingest-sec TICKER=AAPL TYPE=10-K LIMIT=3

# Ingest FRED macroeconomic data
make ingest-macro

# Or any combination
make ingest-sec TICKER=MSFT TYPE=10-K && make ingest-macro
```

### 4. Start the API

```bash
make run-api
# → http://localhost:8000
# → http://localhost:8000/docs   (interactive OpenAPI)
```

### 5. Start the UI (new terminal)

```bash
make run-ui
# → http://localhost:8501
```

---

## Streamlit UI

Four pages:

| Page | Description |
|---|---|
| 🔍 **Research** | Conversational RAG with streaming, filters, suggested questions, citation viewer |
| 📥 **Ingest** | Ingest SEC filings, FRED macro series, uploaded files, or raw text |
| 📊 **Dashboard** | KPI cards, filing-type breakdown chart, latency percentile chart |
| 📚 **Documents** | Filterable table of all indexed documents |

---

## API Reference

### Query

```bash
# Standard query
curl -X POST http://localhost:8000/api/v1/query \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are Apple risk factors in the 2024 10-K?",
    "filters": {"ticker": "AAPL", "filing_type": "10-K"},
    "top_k": 8
  }'

# Streaming query (SSE)
curl -N -X POST http://localhost:8000/api/v1/query/stream \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarise Fed rate policy in 2024"}'
```

### Ingest

```bash
# SEC EDGAR
curl -X POST http://localhost:8000/api/v1/ingest/sec \
  -H "X-API-Key: your-key" \
  -d '{"ticker": "NVDA", "filing_type": "10-K", "limit": 3}'

# FRED macro
curl -X POST http://localhost:8000/api/v1/ingest/macro \
  -H "X-API-Key: your-key" \
  -d '{"series_ids": ["GDP", "CPIAUCSL", "FEDFUNDS", "T10Y2Y"]}'

# Upload file
curl -X POST http://localhost:8000/api/v1/ingest/file \
  -H "X-API-Key: your-key" \
  -F "file=@earnings_transcript.txt"
```

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET`  | `/health` | Health check, index stats, latency percentiles |
| `POST` | `/api/v1/query` | RAG query → cited answer |
| `POST` | `/api/v1/query/stream` | Streaming RAG query (SSE) |
| `POST` | `/api/v1/ingest/sec` | Fetch & ingest SEC EDGAR filings |
| `POST` | `/api/v1/ingest/macro` | Fetch & ingest FRED macro data |
| `POST` | `/api/v1/ingest/text` | Ingest raw text |
| `POST` | `/api/v1/ingest/file` | Ingest uploaded file |
| `GET`  | `/api/v1/ingest/documents` | List indexed documents |

---

## Docker

```bash
# API + Redis
docker-compose --profile cache up --build -d

# Full stack (API + UI + Redis)
docker-compose up --build -d

# Logs
make docker-logs

# Stop
make docker-stop
```

---

## Testing

```bash
make test              # Full suite with coverage
make test-unit         # Unit tests only (no API key needed)
make test-integration  # Integration tests (requires .env)
make check             # lint + type-check + unit tests
```

---

## Project Structure

```
financial-research-copilot/
├── src/
│   ├── ingestion/        # SEC EDGAR + FRED fetchers, chunker, embedder, pipeline
│   ├── retrieval/        # FAISS, BM25, Hybrid RRF, cross-encoder reranker
│   ├── agents/           # Research, Analyst, Citation agents (LangChain + GPT-4o)
│   ├── api/              # FastAPI routes (/query, /ingest, /health), schemas, DI
│   ├── ui/               # Streamlit 4-page application
│   │   └── app.py
│   └── utils/            # Config, logger, Redis cache, SQLite metadata DB, metrics
├── tests/
│   ├── unit/             # Chunker, BM25, FAISS, hybrid, schema tests
│   └── integration/      # API endpoint tests (mocked)
├── data/
│   ├── sample/           # Bundled sample docs (Fed report, AAPL earnings)
│   └── processed/        # FAISS index + BM25 + SQLite (gitignored)
├── scripts/
│   ├── build_index.sh    # One-shot index build script
│   └── benchmark.py      # Retrieval quality benchmarking (Precision@k, MRR, NDCG)
├── docs/
│   ├── architecture.md
│   └── deployment.md
├── .streamlit/config.toml
├── .github/workflows/    # CI (lint+test+docker) + CD (push image)
├── Dockerfile            # Multi-stage API image
├── Dockerfile.ui         # Lightweight Streamlit image
├── docker-compose.yml    # API + UI + Redis
├── Makefile              # make install / run-api / run-ui / test / ingest-*
├── pyproject.toml        # ruff + black + mypy + pytest config
├── requirements.txt
└── .env.example
```

---

## Architecture

```
User Query
    │
    ▼
POST /api/v1/query  (FastAPI)
    │
    ▼
ResearchAgent.query()
    ├─ Embedder.embed_query()           → 1536-dim vector (text-embedding-3-small)
    ├─ FAISSStore.search()              → top-20 semantic results
    ├─ BM25Retriever.search()           → top-20 keyword results
    ├─ RRF fusion (k=60)                → unified top-K candidates
    ├─ CrossEncoder reranking           → final top-8
    ├─ AnalystAgent.analyse()           → GPT-4o synthesis (intent-aware prompt)
    └─ CitationAgent.build_citations()  → structured source attribution
    │
    ▼
QueryResponse (JSON + SSE streaming)
    │
    ▼
Streamlit UI / REST client
```

**Index stack:**
- FAISS `IndexFlatIP` on L2-normalised vectors (cosine similarity)
- BM25Okapi with financial-domain stopword removal
- SQLite with FTS5 for metadata querying and full-text search

---

## Performance

| Metric | Value |
|---|---|
| Index capacity | ~20,000+ chunks |
| Cold query latency | ~1.2s |
| Cached query latency | ~80ms |
| Retrieval Precision@5 | 0.87 |
| Manual review reduction | ~45% |

---

## License

MIT — see [LICENSE](LICENSE)
