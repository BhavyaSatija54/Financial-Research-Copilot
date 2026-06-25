"""Unit tests for API Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    IngestSECRequest,
    IngestTextRequest,
    QueryFiltersSchema,
    QueryRequest,
)


class TestQueryFiltersSchema:
    def test_ticker_uppercased(self) -> None:
        f = QueryFiltersSchema(ticker="aapl")
        assert f.ticker == "AAPL"

    def test_filing_type_uppercased(self) -> None:
        f = QueryFiltersSchema(filing_type="10-k")
        assert f.filing_type == "10-K"

    def test_all_optional(self) -> None:
        f = QueryFiltersSchema()
        assert f.ticker is None
        assert f.company is None

    def test_none_ticker(self) -> None:
        f = QueryFiltersSchema(ticker=None)
        assert f.ticker is None


class TestQueryRequest:
    def test_valid(self) -> None:
        q = QueryRequest(question="What are Apple's key risk factors?")
        assert q.stream is False
        assert q.top_k is None

    def test_question_too_short(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(question="Hi")

    def test_question_too_long(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(question="x" * 2001)

    def test_top_k_bounds(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(question="Valid question here?", top_k=0)
        with pytest.raises(ValidationError):
            QueryRequest(question="Valid question here?", top_k=21)

    def test_with_filters(self) -> None:
        q = QueryRequest(
            question="Summarise revenue trends for Apple",
            filters=QueryFiltersSchema(ticker="AAPL", filing_type="10-K"),
        )
        assert q.filters is not None
        assert q.filters.ticker == "AAPL"


class TestIngestTextRequest:
    def test_valid(self) -> None:
        req = IngestTextRequest(
            doc_id="AAPL_10K_2024",
            text="x" * 100,
            metadata={"ticker": "AAPL"},
        )
        assert req.doc_id == "AAPL_10K_2024"

    def test_text_too_short(self) -> None:
        with pytest.raises(ValidationError):
            IngestTextRequest(doc_id="test", text="short")

    def test_empty_metadata_default(self) -> None:
        req = IngestTextRequest(doc_id="doc", text="x" * 60)
        assert req.metadata == {}


class TestIngestSECRequest:
    def test_ticker_uppercased(self) -> None:
        req = IngestSECRequest(ticker="aapl")
        assert req.ticker == "AAPL"

    def test_default_filing_type(self) -> None:
        req = IngestSECRequest(ticker="MSFT")
        assert req.filing_type == "10-K"

    def test_invalid_filing_type(self) -> None:
        with pytest.raises(ValidationError):
            IngestSECRequest(ticker="AAPL", filing_type="INVALID")

    def test_limit_bounds(self) -> None:
        with pytest.raises(ValidationError):
            IngestSECRequest(ticker="AAPL", limit=0)
        with pytest.raises(ValidationError):
            IngestSECRequest(ticker="AAPL", limit=11)
