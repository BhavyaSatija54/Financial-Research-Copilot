"""
Integration tests for FastAPI endpoints.

These tests use a fully wired app with mocked LLM and embedding calls
so no real API keys are needed in CI.

Mark: integration (run with pytest tests/integration/ -v)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.retrieval.faiss_store import FAISSStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.agents.research_agent import ResearchResponse


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_faiss() -> FAISSStore:
    return FAISSStore(dimension=8)


@pytest.fixture(scope="module")
def mock_bm25() -> BM25Retriever:
    return BM25Retriever()


@pytest.fixture(scope="module")
def mock_response() -> ResearchResponse:
    return ResearchResponse(
        question="What are Apple's risk factors?",
        answer="Apple faces risks including supply chain disruptions and regulatory changes.",
        citations=[
            {
                "citation_number": 1,
                "chunk_id": "abc123",
                "doc_id": "AAPL_10-K_2024",
                "company": "Apple Inc.",
                "ticker": "AAPL",
                "filing_type": "10-K",
                "period": "2024-09-30",
                "filed_date": "2024-11-01",
                "source_url": None,
                "excerpt": "Supply chain risks remain elevated...",
                "relevance_score": 0.92,
                "retrieval_sources": ["semantic", "bm25"],
            }
        ],
        retrieved_chunks=5,
        filters_applied={},
        latency_ms=450.0,
        model="gpt-4o",
    )


@pytest.fixture(scope="module")
def client(mock_faiss: FAISSStore, mock_bm25: BM25Retriever, mock_response: ResearchResponse):
    """TestClient with mocked services."""
    app = create_app()

    async def mock_init(settings: Any) -> None:
        import src.api.dependencies as deps
        deps._faiss_store = mock_faiss
        deps._bm25_retriever = mock_bm25
        deps._embedder = MagicMock()
        deps._cache_client = None

        agent_mock = AsyncMock()
        agent_mock.query = AsyncMock(return_value=mock_response)
        deps._research_agent = agent_mock

        pipeline_mock = AsyncMock()
        pipeline_mock.ingest_text = AsyncMock(return_value=MagicMock(
            status="success", chunks_added=42, tokens_indexed=18000, error=None
        ))
        deps._ingestion_pipeline = pipeline_mock

    async def mock_shutdown() -> None:
        pass

    with (
        patch("src.api.main.initialise_services", side_effect=mock_init),
        patch("src.api.main.shutdown_services", side_effect=mock_shutdown),
    ):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


API_KEY = "change-me"
HEADERS = {"X-API-Key": API_KEY}


# ── Health ────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestHealthEndpoint:
    def test_health_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "index_stats" in data

    def test_root_returns_service_info(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Financial Research Copilot" in resp.json()["service"]


# ── Query ─────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestQueryEndpoint:
    def test_query_success(self, client: TestClient) -> None:
        resp = client.post(
            "/query",
            json={"question": "What are Apple's key risk factors?"},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "citations" in data
        assert data["retrieved_chunks"] >= 0

    def test_query_with_filters(self, client: TestClient) -> None:
        resp = client.post(
            "/query",
            json={
                "question": "Summarise revenue trends for AAPL",
                "filters": {"ticker": "AAPL", "filing_type": "10-K"},
            },
            headers=HEADERS,
        )
        assert resp.status_code == 200

    def test_query_missing_api_key(self, client: TestClient) -> None:
        resp = client.post(
            "/query",
            json={"question": "What are the risk factors?"},
        )
        assert resp.status_code == 401

    def test_query_wrong_api_key(self, client: TestClient) -> None:
        resp = client.post(
            "/query",
            json={"question": "What are the risk factors?"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_query_short_question_rejected(self, client: TestClient) -> None:
        resp = client.post(
            "/query",
            json={"question": "Hi"},
            headers=HEADERS,
        )
        assert resp.status_code == 422


# ── Ingest ────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestIngestEndpoint:
    def test_ingest_text(self, client: TestClient) -> None:
        resp = client.post(
            "/ingest/text",
            json={
                "doc_id": "TEST_DOC_001",
                "text": "Apple Inc. quarterly earnings report. " * 10,
                "metadata": {"ticker": "AAPL"},
            },
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["chunks_added"] > 0

    def test_ingest_missing_auth(self, client: TestClient) -> None:
        resp = client.post(
            "/ingest/text",
            json={"doc_id": "X", "text": "y" * 100},
        )
        assert resp.status_code == 401
