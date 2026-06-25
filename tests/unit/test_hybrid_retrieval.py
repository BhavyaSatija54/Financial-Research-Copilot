"""Unit tests for BM25Retriever, FAISSStore, and HybridRetriever."""

from __future__ import annotations

import numpy as np
import pytest

from src.ingestion.chunker import TextChunk
from src.retrieval.bm25_retriever import BM25Retriever, _tokenise
from src.retrieval.faiss_store import FAISSStore
from src.retrieval.hybrid import HybridRetriever, build_filters


# ── Helpers ───────────────────────────────────────────────────────────

def _make_chunk(
    text: str,
    doc_id: str = "doc1",
    ticker: str = "AAPL",
    filing_type: str = "10-K",
    chunk_index: int = 0,
) -> TextChunk:
    import uuid
    return TextChunk(
        chunk_id=str(uuid.uuid4()),
        text=text,
        token_count=len(text.split()),
        char_start=0,
        char_end=len(text),
        chunk_index=chunk_index,
        total_chunks=1,
        metadata={"doc_id": doc_id, "ticker": ticker, "filing_type": filing_type},
    )


def _make_embedding(dim: int = 8) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ── Tokeniser ─────────────────────────────────────────────────────────

class TestTokenise:
    def test_removes_stopwords(self) -> None:
        tokens = _tokenise("the revenue is growing and profits are up")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "revenue" in tokens

    def test_lowercases(self) -> None:
        tokens = _tokenise("Apple AAPL Revenue")
        assert all(t == t.lower() for t in tokens)

    def test_filters_short(self) -> None:
        tokens = _tokenise("a b c revenue")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "revenue" in tokens

    def test_empty(self) -> None:
        assert _tokenise("") == []


# ── BM25Retriever ─────────────────────────────────────────────────────

class TestBM25Retriever:
    @pytest.fixture
    def retriever(self) -> BM25Retriever:
        r = BM25Retriever()
        chunks = [
            _make_chunk("Apple revenue increased by 15% year-over-year in Q4", ticker="AAPL"),
            _make_chunk("Microsoft cloud business showed strong growth metrics", ticker="MSFT"),
            _make_chunk("Apple faces supply chain risks and geopolitical headwinds", ticker="AAPL"),
            _make_chunk("Federal Reserve interest rate decisions impact consumer spending", ticker="MACRO"),
        ]
        r.add_chunks(chunks)
        return r

    def test_search_returns_results(self, retriever: BM25Retriever) -> None:
        results = retriever.search("Apple revenue", top_k=3)
        assert len(results) > 0

    def test_top_result_relevant(self, retriever: BM25Retriever) -> None:
        results = retriever.search("Apple revenue Q4", top_k=3)
        top_text = results[0].text.lower()
        assert "apple" in top_text or "revenue" in top_text

    def test_filter_by_ticker(self, retriever: BM25Retriever) -> None:
        results = retriever.search("business growth", top_k=5, filters={"ticker": "MSFT"})
        assert all(r.metadata["ticker"] == "MSFT" for r in results)

    def test_no_results_on_unmatched_filter(self, retriever: BM25Retriever) -> None:
        results = retriever.search("revenue", top_k=5, filters={"ticker": "TSLA"})
        assert results == []

    def test_total_chunks(self, retriever: BM25Retriever) -> None:
        assert retriever.total_chunks == 4

    def test_scores_positive(self, retriever: BM25Retriever) -> None:
        results = retriever.search("Apple revenue", top_k=3)
        assert all(r.score > 0 for r in results)

    def test_ranks_sequential(self, retriever: BM25Retriever) -> None:
        results = retriever.search("Apple revenue", top_k=3)
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_empty_query_returns_empty(self, retriever: BM25Retriever) -> None:
        # Only stopwords — all filtered out
        results = retriever.search("the a and", top_k=3)
        assert results == []


# ── FAISSStore ────────────────────────────────────────────────────────

class TestFAISSStore:
    DIM = 8

    @pytest.fixture
    def store(self) -> FAISSStore:
        s = FAISSStore(dimension=self.DIM)
        chunks = [
            _make_chunk("Apple quarterly earnings beat expectations"),
            _make_chunk("Risk factors related to competition"),
        ]
        embeddings = np.stack([_make_embedding(self.DIM) for _ in chunks])
        s.add_chunks(chunks, embeddings)
        return s

    def test_total_vectors(self, store: FAISSStore) -> None:
        assert store.total_vectors == 2

    def test_search_returns_results(self, store: FAISSStore) -> None:
        query = _make_embedding(self.DIM)
        results = store.search(query, top_k=2)
        assert len(results) == 2

    def test_scores_in_valid_range(self, store: FAISSStore) -> None:
        query = _make_embedding(self.DIM)
        results = store.search(query, top_k=2)
        for r in results:
            assert -1.0 <= r.score <= 1.0 + 1e-5

    def test_filter_by_ticker(self, store: FAISSStore) -> None:
        query = _make_embedding(self.DIM)
        results = store.search(query, top_k=10, filters={"ticker": "AAPL"})
        assert all(r.metadata["ticker"] == "AAPL" for r in results)

    def test_empty_store_returns_empty(self) -> None:
        s = FAISSStore(dimension=self.DIM)
        results = s.search(_make_embedding(self.DIM), top_k=5)
        assert results == []


# ── HybridRetriever ───────────────────────────────────────────────────

class TestHybridRetriever:
    DIM = 8

    @pytest.fixture
    def hybrid(self) -> HybridRetriever:
        import uuid
        chunks = [
            _make_chunk("Apple revenue increased significantly in Q4 2024", ticker="AAPL"),
            _make_chunk("Risk factors include supply chain disruptions", ticker="AAPL"),
            _make_chunk("Microsoft Azure grew 29% year over year", ticker="MSFT"),
        ]
        embeddings = np.stack([_make_embedding(self.DIM) for _ in chunks])

        faiss = FAISSStore(dimension=self.DIM)
        faiss.add_chunks(chunks, embeddings)

        bm25 = BM25Retriever()
        bm25.add_chunks(chunks)

        return HybridRetriever(
            faiss_store=faiss,
            bm25_retriever=bm25,
            rrf_k=60,
            top_k_semantic=10,
            top_k_bm25=10,
            top_k_final=5,
        )

    def test_returns_results(self, hybrid: HybridRetriever) -> None:
        query_vec = _make_embedding(self.DIM)
        results = hybrid.retrieve("Apple quarterly revenue", query_vec)
        assert len(results) > 0

    def test_rrf_scores_positive(self, hybrid: HybridRetriever) -> None:
        query_vec = _make_embedding(self.DIM)
        results = hybrid.retrieve("Apple revenue", query_vec)
        assert all(r.rrf_score > 0 for r in results)

    def test_results_sorted_by_rrf(self, hybrid: HybridRetriever) -> None:
        query_vec = _make_embedding(self.DIM)
        results = hybrid.retrieve("revenue", query_vec)
        scores = [r.rrf_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_filter_applied(self, hybrid: HybridRetriever) -> None:
        query_vec = _make_embedding(self.DIM)
        results = hybrid.retrieve("revenue", query_vec, filters={"ticker": "MSFT"})
        assert all(r.metadata["ticker"] == "MSFT" for r in results)


# ── build_filters helper ─────────────────────────────────────────────

class TestBuildFilters:
    def test_none_when_empty(self) -> None:
        assert build_filters() is None

    def test_ticker_uppercased(self) -> None:
        f = build_filters(ticker="aapl")
        assert f is not None
        assert f["ticker"] == "AAPL"

    def test_multiple_filters(self) -> None:
        f = build_filters(ticker="AAPL", filing_type="10-K", doc_type="sec")
        assert f is not None
        assert len(f) == 3
