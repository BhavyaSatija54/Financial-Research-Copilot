# Architecture

## Overview

Financial Research Copilot is a **hybrid Retrieval-Augmented Generation (RAG)** system designed for high-fidelity financial document analysis. It combines dense semantic retrieval (FAISS), sparse keyword retrieval (BM25), contextual reranking, and LLM-based synthesis into a production-grade REST API.

---

## Components

### 1. Ingestion Pipeline (`src/ingestion/`)

| Module | Role |
|---|---|
| `sec_fetcher.py` | Async SEC EDGAR client; fetches 10-K/10-Q/8-K by ticker |
| `chunker.py` | Sentence-aware sliding-window tokeniser; targets 512 tokens/chunk with 64-token overlap |
| `embedder.py` | Batched OpenAI `text-embedding-3-small` with retry logic |
| `pipeline.py` | Orchestrates chunking → embedding → FAISS + BM25 indexing in parallel |

**Chunking strategy:** Sentences are split using regex sentence boundaries that respect financial section headers (ITEM 1A, PART II, etc.). Chunks accumulate sentences until the token budget is reached, then the window slides forward with overlap to preserve context across boundaries.

### 2. Vector Store (`src/retrieval/faiss_store.py`)

- **Index type:** `IndexFlatIP` (flat inner-product) on L2-normalised vectors = cosine similarity
- **Metadata side-store:** Parallel `dict` mapping FAISS integer IDs → `TextChunk` objects with full metadata
- **Filtered search:** Post-ANN metadata filtering; over-fetches by 10× to fill the top-k after filtering
- **Scale path:** For >1M vectors, swap `IndexFlatIP` with `IndexIVFFlat` or `IndexHNSWFlat`

### 3. BM25 Retriever (`src/retrieval/bm25_retriever.py`)

- **Algorithm:** BM25Okapi (rank-bm25)
- **Tokenisation:** Lowercase, strip punctuation, remove financial stopwords
- **Rebuild strategy:** Lazy rebuild on first search after incremental adds

### 4. Hybrid Retrieval (`src/retrieval/hybrid.py`)

Reciprocal Rank Fusion merges FAISS and BM25 ranked lists:

```
RRF(d) = Σ_i  1 / (k + rank_i(d))     k = 60
```

Both lists contribute equally by default. RRF is robust to score-scale mismatches between dense and sparse retrievers and consistently outperforms linear combination in practice.

### 5. Reranker (`src/retrieval/reranker.py`)

- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (6-layer BERT cross-encoder)
- **Input:** Top-K (query, passage) pairs from hybrid retrieval
- **Fallback:** If sentence-transformers is not installed, falls back to RRF score ordering
- **Latency:** ~150ms for 20 candidates on CPU; ~30ms on GPU

### 6. Agent Layer (`src/agents/`)

```
ResearchAgent
  ├── embed query
  ├── HybridRetriever.retrieve()
  ├── Reranker.rerank()
  ├── AnalystAgent.analyse()   ← GPT-4o synthesis
  └── CitationAgent.build_citations()
```

The **AnalystAgent** uses a system prompt enforcing source-only faithfulness. Intent classification (risk, financial_metrics, guidance, macro, etc.) adjusts the synthesis style.

### 7. API Layer (`src/api/`)

- **Framework:** FastAPI with async endpoints
- **Auth:** API key via `X-API-Key` header
- **Rate limiting:** slowapi, 60 req/min per IP
- **Streaming:** SSE via `StreamingResponse`
- **Caching:** Redis with SHA-256 keyed on (question, filters)

---

## Data Flow

```
User Query
    │
    ▼
POST /query (FastAPI)
    │
    ▼
ResearchAgent.query()
    ├── Embedder.embed_query()          → query_vector (1536d)
    ├── HybridRetriever.retrieve()
    │       ├── FAISSStore.search()     → top-20 semantic
    │       ├── BM25Retriever.search()  → top-20 keyword
    │       └── RRF fusion             → top-8 candidates
    ├── Reranker.rerank()              → final top-5
    ├── AnalystAgent.analyse()         → GPT-4o answer
    └── CitationAgent.build_citations() → structured sources
    │
    ▼
QueryResponse (JSON)
```

---

## Deployment

See [deployment.md](deployment.md) for Docker, Kubernetes, and cloud deployment guides.
