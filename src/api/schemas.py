"""
Pydantic v2 request/response schemas for the FastAPI layer.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ── Query ─────────────────────────────────────────────────────────────

class QueryFiltersSchema(BaseModel):
    ticker: str | None = Field(None, example="AAPL", description="Stock ticker symbol")
    company: str | None = Field(None, example="Apple Inc.")
    filing_type: str | None = Field(
        None,
        example="10-K",
        description="SEC filing type: 10-K, 10-Q, 8-K, etc.",
    )
    date_from: str | None = Field(None, example="2022-01-01", description="ISO date string")
    date_to: str | None = Field(None, example="2024-12-31")
    doc_type: str | None = Field(None, description="Custom doc type: earnings, macro, etc.")

    @field_validator("ticker", mode="before")
    @classmethod
    def upper_ticker(cls, v: str | None) -> str | None:
        return v.upper().strip() if v else v

    @field_validator("filing_type", mode="before")
    @classmethod
    def upper_filing(cls, v: str | None) -> str | None:
        return v.upper().strip() if v else v


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        example="What are Apple's key risk factors in the 2024 10-K?",
    )
    filters: QueryFiltersSchema | None = None
    top_k: int | None = Field(None, ge=1, le=20, description="Override number of retrieved chunks")
    stream: bool = Field(False, description="Enable SSE streaming response")


class CitationSchema(BaseModel):
    citation_number: int
    chunk_id: str
    doc_id: str
    company: str
    ticker: str | None
    filing_type: str
    period: str | None
    filed_date: str | None
    source_url: str | None
    excerpt: str
    relevance_score: float
    retrieval_sources: list[str]


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationSchema]
    retrieved_chunks: int
    filters_applied: dict[str, Any]
    latency_ms: float
    model: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Ingestion ────────────────────────────────────────────────────────

class IngestTextRequest(BaseModel):
    doc_id: str = Field(..., example="AAPL_10K_2024", description="Unique document identifier")
    text: str = Field(..., min_length=50, description="Raw document text")
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestSECRequest(BaseModel):
    ticker: str = Field(..., example="AAPL")
    filing_type: Literal["10-K", "10-Q", "8-K"] = "10-K"
    limit: int = Field(3, ge=1, le=10, description="Max filings to fetch")

    @field_validator("ticker", mode="before")
    @classmethod
    def upper_ticker(cls, v: str) -> str:
        return v.upper().strip()


class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    chunks_added: int
    tokens_indexed: int
    errors: list[str] = Field(default_factory=list)


# ── Health ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    version: str
    environment: str
    index_stats: dict[str, Any]
    cache_available: bool
    latency_stats: dict[str, Any] = Field(default_factory=dict)


# ── Error ─────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    request_id: str | None = None
