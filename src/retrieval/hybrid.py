"""
Hybrid retrieval engine combining FAISS semantic search and BM25 keyword
search via Reciprocal Rank Fusion (RRF).

RRF score = Σ  1 / (k + rank_i)
            i

Where k is a smoothing constant (typically 60) and rank_i is the rank
of the document in retrieval list i.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.retrieval.bm25_retriever import BM25Retriever, BM25Result
from src.retrieval.faiss_store import FAISSStore, SearchResult
from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.utils.metrics import track_latency

log = get_logger(__name__)


@dataclass
class HybridResult:
    """Unified result after RRF fusion."""

    chunk_id: str
    text: str
    rrf_score: float
    semantic_score: float | None
    bm25_score: float | None
    metadata: dict[str, Any]
    rank: int = 0
    sources: list[str] = field(default_factory=list)  # ["semantic", "bm25"]


class HybridRetriever:
    """
    Combines FAISS + BM25 with Reciprocal Rank Fusion.

    Parameters
    ----------
    faiss_store : FAISSStore
    bm25_retriever : BM25Retriever
    rrf_k : int
        RRF smoothing constant (default 60 from settings).
    top_k_semantic : int
        Candidates fetched from FAISS per query.
    top_k_bm25 : int
        Candidates fetched from BM25 per query.
    top_k_final : int
        Final results returned after fusion.
    """

    def __init__(
        self,
        faiss_store: FAISSStore,
        bm25_retriever: BM25Retriever,
        rrf_k: int | None = None,
        top_k_semantic: int | None = None,
        top_k_bm25: int | None = None,
        top_k_final: int | None = None,
    ) -> None:
        cfg = get_settings()
        self.faiss = faiss_store
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k or cfg.rrf_k
        self.top_k_semantic = top_k_semantic or cfg.top_k_semantic
        self.top_k_bm25 = top_k_bm25 or cfg.top_k_bm25
        self.top_k_final = top_k_final or cfg.top_k_reranked

    def retrieve(
        self,
        query: str,
        query_vector: np.ndarray,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[HybridResult]:
        """
        Run hybrid retrieval.

        Parameters
        ----------
        query : str
            Raw query string (for BM25).
        query_vector : np.ndarray
            Pre-computed query embedding (for FAISS).
        top_k : int | None
            Override for final result count.
        filters : dict | None
            Metadata filters applied to both retrievers.

        Returns
        -------
        list[HybridResult]  sorted by descending RRF score
        """
        k = top_k or self.top_k_final

        with track_latency("hybrid_retrieve", query_len=len(query), top_k=k):
            semantic_results: list[SearchResult] = self.faiss.search(
                query_vector,
                top_k=self.top_k_semantic,
                filters=filters,
            )
            bm25_results: list[BM25Result] = self.bm25.search(
                query,
                top_k=self.top_k_bm25,
                filters=filters,
            )

        fused = self._reciprocal_rank_fusion(semantic_results, bm25_results, k)

        log.info(
            "hybrid_retrieve_done",
            semantic_hits=len(semantic_results),
            bm25_hits=len(bm25_results),
            fused=len(fused),
            filters=filters,
        )
        return fused

    # ── RRF ──────────────────────────────────────────────────────────

    def _reciprocal_rank_fusion(
        self,
        semantic: list[SearchResult],
        bm25: list[BM25Result],
        top_k: int,
    ) -> list[HybridResult]:
        """Merge two ranked lists using Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        semantic_scores: dict[str, float] = {}
        bm25_scores: dict[str, float] = {}
        chunk_data: dict[str, dict] = {}

        # Process semantic results
        for rank, result in enumerate(semantic, start=1):
            cid = result.chunk_id
            rrf = 1.0 / (self.rrf_k + rank)
            scores[cid] = scores.get(cid, 0.0) + rrf
            semantic_scores[cid] = result.score
            chunk_data[cid] = {"text": result.text, "metadata": result.metadata}

        # Process BM25 results
        for rank, result in enumerate(bm25, start=1):
            cid = result.chunk_id
            rrf = 1.0 / (self.rrf_k + rank)
            scores[cid] = scores.get(cid, 0.0) + rrf
            bm25_scores[cid] = result.score
            if cid not in chunk_data:
                chunk_data[cid] = {"text": result.text, "metadata": result.metadata}

        # Build sorted output
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results: list[HybridResult] = []
        for rank, (cid, rrf_score) in enumerate(ranked, start=1):
            sources = []
            if cid in semantic_scores:
                sources.append("semantic")
            if cid in bm25_scores:
                sources.append("bm25")

            results.append(
                HybridResult(
                    chunk_id=cid,
                    text=chunk_data[cid]["text"],
                    rrf_score=rrf_score,
                    semantic_score=semantic_scores.get(cid),
                    bm25_score=bm25_scores.get(cid),
                    metadata=chunk_data[cid]["metadata"],
                    rank=rank,
                    sources=sources,
                )
            )

        return results


def build_filters(
    ticker: str | None = None,
    company: str | None = None,
    filing_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    doc_type: str | None = None,
    extra: dict | None = None,
) -> dict[str, Any] | None:
    """
    Convenience builder for metadata filters.

    Returns None if no filters are set (skips filtering entirely).
    """
    filters: dict[str, Any] = {}

    if ticker:
        filters["ticker"] = ticker.upper()
    if company:
        filters["company"] = company
    if filing_type:
        filters["filing_type"] = filing_type
    if doc_type:
        filters["doc_type"] = doc_type
    if extra:
        filters.update(extra)

    # Date range filtering is done post-retrieval because FAISS
    # doesn't natively support range predicates.
    # (date_from / date_to are handled in the agent layer)

    return filters if filters else None
