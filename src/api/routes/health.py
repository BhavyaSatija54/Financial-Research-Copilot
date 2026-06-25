"""
GET /health  — liveness + readiness probe.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from typing import Annotated

from src.api.dependencies import get_cache, get_faiss_store, get_bm25_retriever
from src.api.schemas import HealthResponse
from src.retrieval.faiss_store import FAISSStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.utils.cache import CacheClient
from src.utils.config import get_settings
from src.utils.metrics import get_all_latency_stats

router = APIRouter(tags=["Health"])

_VERSION = "1.0.0"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Returns index statistics, cache status, and latency percentiles.",
)
async def health(
    faiss: Annotated[FAISSStore, Depends(get_faiss_store)],
    bm25: Annotated[BM25Retriever, Depends(get_bm25_retriever)],
    cache: Annotated[CacheClient | None, Depends(get_cache)],
) -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=_VERSION,
        environment=settings.environment,
        index_stats={
            "faiss_vectors": faiss.total_vectors,
            "bm25_chunks": bm25.total_chunks,
            "embedding_model": settings.embedding_model,
            "llm_model": settings.openai_model,
        },
        cache_available=cache.available if cache else False,
        latency_stats=get_all_latency_stats(),
    )


@router.get("/", include_in_schema=False)
async def root() -> dict:
    return {"service": "Financial Research Copilot", "version": _VERSION, "docs": "/docs"}
