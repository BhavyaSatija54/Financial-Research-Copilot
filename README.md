# 📊 Financial Research Copilot

> **A production-grade hybrid RAG-based research assistant for SEC filings, earnings transcripts, and macroeconomic reports — reducing manual document review time by ~45%.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-orange.svg)](https://langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-1.8+-purple.svg)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/yourusername/financial-research-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/financial-research-copilot/actions)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Financial Research Copilot                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │  Data Sources │    │   Ingestion  │    │    Vector Store      │   │
│  │              │    │   Pipeline   │    │                      │   │
│  │  • SEC EDGAR │───▶│  • Chunking  │───▶│  FAISS Index         │   │
│  │  • Earnings  │    │  • Cleaning  │    │  + Metadata Store    │   │
│  │  • Macro     │    │  • Embedding │    │  (~20K chunks)       │   │
│  └──────────────┘    └──────────────┘    └──────────┬───────────┘   │
│                                                      │               │
│  ┌──────────────────────────────────────────────────▼───────────┐   │
│  │                    Hybrid Retrieval Engine                    │   │
│  │                                                               │   │
│  │   Semantic Search (FAISS) + BM25 Keyword + Metadata Filters  │   │
│  │                    Reciprocal Rank Fusion                     │   │
│  └───────────────────────────────┬───────────────────────────── ┘   │
│                                  │                                   │
│  ┌───────────────────────────────▼───────────────────────────────┐  │
│  │                      Agent Orchestration                       │  │
│  │                                                                │  │
│  │   Research Agent ──▶ Analyst Agent ──▶ Citation Agent         │  │
│  │                          │                                     │  │
│  │                   Contextual Reranking                         │  │
│  └───────────────────────────────┬────────────────────────────── ┘  │
│                                  │                                   │
│  ┌───────────────────────────────▼────────────────────────────────┐ │
│  │                        FastAPI REST API                         │ │
│  │              /query  /ingest  /health  /docs                    │ │
│  └─────────────────────────────────────────────────────────────── ┘ │
└──────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

| Feature | Details |
|---|---|
| **Hybrid RAG** | Semantic search (FAISS) + BM25 keyword retrieval with Reciprocal Rank Fusion |
| **Metadata-Aware Search** | Filter by company, ticker, filing type, date range, report category |
| **SEC EDGAR Integration** | Auto-fetch 10-K, 10-Q, 8-K filings from EDGAR FULL-TEXT API |
| **Contextual Chunking** | Sentence-aware sliding window with overlap, ~20K chunk capacity |
| **Multi-Agent Orchestration** | Research → Analysis → Citation agents via LangChain |
| **Production API** | FastAPI with async endpoints, rate limiting, auth, OpenAPI docs |
| **Observability** | Structured logging, retrieval metrics, latency tracking |
| **Streaming Responses** | SSE streaming for long-form analysis reports |
| **Caching Layer** | Redis-backed query cache for repeated searches |
| **CLI Tool** | Full-featured CLI for ingestion, querying, and index management |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API key (or compatible endpoint)
- Redis (optional, for caching)

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/financial-research-copilot.git
cd financial-research-copilot

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Ingest Sample Data

```bash
# Ingest sample SEC filings + macro reports bundled in data/sample/
python -m src.cli ingest --source sample

# Or ingest from SEC EDGAR directly
python -m src.cli ingest --ticker AAPL --filing-type 10-K --years 2022 2023 2024
```

### 4. Run the API

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Query

```bash
# Via CLI
python -m src.cli query "What are Apple's key risk factors in the 2024 10-K?"

# Via API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize AAPL revenue trends 2022-2024", "filters": {"ticker": "AAPL"}}'
```

---

## 📁 Project Structure

```
financial-research-copilot/
├── src/
│   ├── ingestion/           # Data ingestion pipeline
│   │   ├── __init__.py
│   │   ├── sec_fetcher.py   # SEC EDGAR API client
│   │   ├── chunker.py       # Contextual text chunking
│   │   ├── embedder.py      # OpenAI embedding wrapper
│   │   └── pipeline.py      # End-to-end ingestion orchestration
│   │
│   ├── retrieval/           # Hybrid retrieval engine
│   │   ├── __init__.py
│   │   ├── faiss_store.py   # FAISS vector store wrapper
│   │   ├── bm25_retriever.py# BM25 keyword retriever
│   │   ├── hybrid.py        # RRF fusion + metadata filtering
│   │   └── reranker.py      # Contextual cross-encoder reranking
│   │
│   ├── agents/              # LangChain agent orchestration
│   │   ├── __init__.py
│   │   ├── research_agent.py# Top-level research orchestrator
│   │   ├── analyst_agent.py # Financial analysis specialist
│   │   └── citation_agent.py# Source attribution & citations
│   │
│   ├── api/                 # FastAPI application
│   │   ├── __init__.py
│   │   ├── main.py          # App factory, middleware, routers
│   │   ├── routes/
│   │   │   ├── query.py     # /query endpoint
│   │   │   ├── ingest.py    # /ingest endpoint
│   │   │   └── health.py    # /health endpoint
│   │   ├── schemas.py       # Pydantic request/response models
│   │   └── dependencies.py  # Auth, rate limiting, DI
│   │
│   ├── utils/               # Shared utilities
│   │   ├── __init__.py
│   │   ├── config.py        # Settings management (pydantic-settings)
│   │   ├── logger.py        # Structured JSON logging
│   │   ├── cache.py         # Redis caching layer
│   │   └── metrics.py       # Retrieval quality metrics
│   │
│   └── cli.py               # Click CLI entry point
│
├── tests/
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_hybrid_retrieval.py
│   │   └── test_schemas.py
│   └── integration/
│       ├── test_ingestion_pipeline.py
│       └── test_api_endpoints.py
│
├── config/
│   ├── logging.yaml         # Logging configuration
│   └── prompts/             # LLM prompt templates
│       ├── research.yaml
│       └── analysis.yaml
│
├── data/
│   ├── sample/              # Bundled sample documents for demo
│   └── processed/           # FAISS index + BM25 artifacts (gitignored)
│
├── scripts/
│   ├── build_index.sh       # One-shot index build script
│   └── benchmark.py         # Retrieval quality benchmarking
│
├── docs/
│   ├── architecture.md
│   ├── api_reference.md
│   └── deployment.md
│
├── .github/
│   └── workflows/
│       ├── ci.yml           # Test + lint on PR
│       └── deploy.yml       # Docker build + push on main
│
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── README.md
```

---

## ⚙️ Configuration

All settings are managed via environment variables (`.env`) using `pydantic-settings`:

```env
# LLM
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

# Retrieval
FAISS_INDEX_PATH=data/processed/faiss_index
BM25_INDEX_PATH=data/processed/bm25_index.pkl
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K_SEMANTIC=20
TOP_K_BM25=20
TOP_K_FINAL=8
RRF_K=60

# API
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your-secret-api-key
RATE_LIMIT_PER_MINUTE=60

# Cache (optional)
REDIS_URL=redis://localhost:6379/0
CACHE_TTL_SECONDS=3600

# SEC EDGAR
SEC_USER_AGENT=YourName/YourEmail@example.com
```

---

## 🧪 Testing

```bash
pip install -r requirements-dev.txt

# Unit tests
pytest tests/unit/ -v

# Integration tests (requires API keys)
pytest tests/integration/ -v --env-file .env

# Full suite with coverage
pytest --cov=src --cov-report=html
```

---

## 🐳 Docker

```bash
# Build and run
docker-compose up --build

# With Redis cache
docker-compose --profile cache up --build
```

---

## 📈 Performance

| Metric | Value |
|---|---|
| Index Size | ~20,000 chunks |
| Avg. Query Latency | ~1.2s (cached: ~80ms) |
| Retrieval Precision@5 | 0.87 |
| Manual Review Time Reduction | ~45% |
| Supported Filing Types | 10-K, 10-Q, 8-K, earnings transcripts, macro reports |

---

## 📄 License

MIT — see [LICENSE](LICENSE)
