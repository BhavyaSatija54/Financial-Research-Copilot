"""GET /health — liveness + readiness probe."""
from __future__ import annotations
from typing import Annotated

from fastapi import APIRouter, Depends
from src.api.dependencies import get_cache, get_faiss_store, get_bm25_retriever, get_metadata_db
from src.api.schemas import HealthResponse
from src.retrieval.faiss_store import FAISSStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.utils.cache import CacheClient
from src.utils.config import get_settings
from src.utils.metadata_db import MetadataDB
from src.utils.metrics import get_all_latency_stats

router = APIRouter(tags=["Health"])
_VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse, summary="Service health check")
async def health(
    faiss: Annotated[FAISSStore, Depends(get_faiss_store)],
    bm25: Annotated[BM25Retriever, Depends(get_bm25_retriever)],
    db: Annotated[MetadataDB, Depends(get_metadata_db)],
    cache: Annotated[CacheClient | None, Depends(get_cache)],
) -> HealthResponse:
    settings = get_settings()
    db_stats = db.stats()
    return HealthResponse(
        status="ok",
        version=_VERSION,
        environment=settings.environment,
        index_stats={
            "faiss_vectors": faiss.total_vectors,
            "bm25_chunks": bm25.total_chunks,
            "sqlite_chunks": db_stats.get("total_chunks", 0),
            "sqlite_documents": db_stats.get("total_documents", 0),
            "sqlite_tickers": db_stats.get("total_tickers", 0),
            "by_filing_type": db_stats.get("by_filing_type", {}),
            "embedding_model": settings.embedding_model,
            "llm_model": settings.openai_model,
        },
        cache_available=cache.available if cache else False,
        latency_stats=get_all_latency_stats(),
    )


@router.get("/", include_in_schema=False)
async def root() -> dict:
    return {"service": "Financial Research Copilot", "version": _VERSION, "docs": "/docs"}
