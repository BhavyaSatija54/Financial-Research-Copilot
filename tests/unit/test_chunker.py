"""Unit tests for the DocumentChunker."""

from __future__ import annotations

import pytest

from src.ingestion.chunker import DocumentChunker, TextChunk, count_tokens, split_sentences


# ── split_sentences ───────────────────────────────────────────────────

class TestSplitSentences:
    def test_basic_split(self) -> None:
        text = "Apple reported revenue of $100B. This was a record quarter. Margins improved."
        sentences = split_sentences(text)
        assert len(sentences) >= 2

    def test_preserves_content(self) -> None:
        text = "Revenue was $50M. EPS was $2.10."
        sentences = split_sentences(text)
        combined = " ".join(sentences)
        assert "Revenue" in combined
        assert "EPS" in combined

    def test_handles_empty(self) -> None:
        assert split_sentences("") == []

    def test_section_header_splits(self) -> None:
        text = "Some text before.\nITEM 1A. Risk Factors\nThis section covers risks."
        sentences = split_sentences(text)
        assert any("ITEM" in s or "Risk" in s for s in sentences)


# ── count_tokens ──────────────────────────────────────────────────────

class TestCountTokens:
    def test_empty_string(self) -> None:
        assert count_tokens("") == 0

    def test_approximate_count(self) -> None:
        # ~4 chars per token rule of thumb
        text = "hello world " * 100
        count = count_tokens(text)
        assert 100 < count < 600  # rough range

    def test_deterministic(self) -> None:
        text = "The quick brown fox"
        assert count_tokens(text) == count_tokens(text)


# ── DocumentChunker ───────────────────────────────────────────────────

class TestDocumentChunker:
    @pytest.fixture
    def chunker(self) -> DocumentChunker:
        return DocumentChunker(chunk_size=100, chunk_overlap=20, min_chunk_length=10)

    @pytest.fixture
    def sample_text(self) -> str:
        return " ".join(
            ["Apple Inc. reported strong quarterly earnings. " * 5] * 20
        )

    def test_returns_chunks(self, chunker: DocumentChunker, sample_text: str) -> None:
        chunks = chunker.chunk_document(sample_text, doc_id="test_doc")
        assert len(chunks) > 0
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_chunk_ids_unique(self, chunker: DocumentChunker, sample_text: str) -> None:
        chunks = chunker.chunk_document(sample_text, doc_id="test_doc")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_metadata_propagated(self, chunker: DocumentChunker, sample_text: str) -> None:
        meta = {"ticker": "AAPL", "filing_type": "10-K"}
        chunks = chunker.chunk_document(sample_text, doc_id="AAPL_10K", metadata=meta)
        for chunk in chunks:
            assert chunk.metadata["ticker"] == "AAPL"
            assert chunk.metadata["filing_type"] == "10-K"
            assert chunk.metadata["doc_id"] == "AAPL_10K"

    def test_token_budget_respected(self, chunker: DocumentChunker, sample_text: str) -> None:
        chunks = chunker.chunk_document(sample_text, doc_id="test_doc")
        # Allow small overrun for edge cases
        assert all(c.token_count <= chunker.chunk_size + 20 for c in chunks)

    def test_total_chunks_backfilled(self, chunker: DocumentChunker, sample_text: str) -> None:
        chunks = chunker.chunk_document(sample_text, doc_id="test_doc")
        total = len(chunks)
        assert all(c.total_chunks == total for c in chunks)

    def test_empty_text_returns_empty(self, chunker: DocumentChunker) -> None:
        chunks = chunker.chunk_document("", doc_id="empty")
        assert chunks == []

    def test_chunk_indices_sequential(self, chunker: DocumentChunker, sample_text: str) -> None:
        chunks = chunker.chunk_document(sample_text, doc_id="test_doc")
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_text_non_empty(self, chunker: DocumentChunker, sample_text: str) -> None:
        chunks = chunker.chunk_document(sample_text, doc_id="test_doc")
        assert all(len(c.text.strip()) > 0 for c in chunks)

    def test_to_dict_keys(self, chunker: DocumentChunker, sample_text: str) -> None:
        chunk = chunker.chunk_document(sample_text, doc_id="test_doc")[0]
        d = chunk.to_dict()
        for key in ("chunk_id", "text", "token_count", "chunk_index", "total_chunks", "doc_id"):
            assert key in d
