"""
FastAPI dependency injection: auth, rate limiting, shared service instances.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.agents.research_agent import ResearchAgent
from src.ingestion.embedder import Embedder
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.faiss_store import FAISSStore
from src.utils.cache import CacheClient
from src.utils.config import Settings, get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── API Key Auth ──────────────────────────────────────────────────────

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: Annotated[str | None, Security(_API_KEY_HEADER)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> str:
    if not api_key or api_key != settings.api_key_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Pass X-API-Key header.",
        )
    return api_key


# ── Singletons (initialised at startup) ─────────────────────────────

_faiss_store: FAISSStore | None = None
_bm25_retriever: BM25Retriever | None = None
_embedder: Embedder | None = None
_cache_client: CacheClient | None = None
_research_agent: ResearchAgent | None = None
_ingestion_pipeline: IngestionPipeline | None = None


def get_faiss_store() -> FAISSStore:
    assert _faiss_store is not None, "FAISSStore not initialised"
    return _faiss_store


def get_bm25_retriever() -> BM25Retriever:
    assert _bm25_retriever is not None, "BM25Retriever not initialised"
    return _bm25_retriever


def get_embedder() -> Embedder:
    assert _embedder is not None, "Embedder not initialised"
    return _embedder


def get_cache() -> CacheClient | None:
    return _cache_client


def get_research_agent() -> ResearchAgent:
    assert _research_agent is not None, "ResearchAgent not initialised"
    return _research_agent


def get_ingestion_pipeline() -> IngestionPipeline:
    assert _ingestion_pipeline is not None, "IngestionPipeline not initialised"
    return _ingestion_pipeline


async def initialise_services(settings: Settings) -> None:
    """Called once at app startup to wire up all singletons."""
    global _faiss_store, _bm25_retriever, _embedder
    global _cache_client, _research_agent, _ingestion_pipeline

    log.info("services_init_start")

    # Vector stores
    _faiss_store = FAISSStore()
    _faiss_store.load()

    _bm25_retriever = BM25Retriever()
    _bm25_retriever.load()

    # Embedder
    _embedder = Embedder()

    # Cache
    if settings.cache_enabled:
        _cache_client = CacheClient(
            url=settings.redis_url,
            ttl=settings.cache_ttl_seconds,
            enabled=True,
        )
        await _cache_client.connect()
    else:
        _cache_client = None

    # Agent
    _research_agent = ResearchAgent(
        faiss_store=_faiss_store,
        bm25_retriever=_bm25_retriever,
        embedder=_embedder,
        cache=_cache_client,
    )

    # Ingestion pipeline
    _ingestion_pipeline = IngestionPipeline(
        faiss_store=_faiss_store,
        bm25_retriever=_bm25_retriever,
        embedder=_embedder,
    )

    log.info(
        "services_init_done",
        faiss_vectors=_faiss_store.total_vectors,
        bm25_chunks=_bm25_retriever.total_chunks,
        cache=_cache_client is not None,
    )


async def shutdown_services() -> None:
    """Called at app shutdown."""
    if _cache_client:
        await _cache_client.disconnect()
    log.info("services_shutdown")
