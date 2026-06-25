"""
Research Agent — top-level query orchestrator.

Workflow:
  1. Parse & classify the query (intent detection)
  2. Extract metadata filters (ticker, filing type, date range)
  3. Embed the query
  4. Run hybrid retrieval
  5. Rerank results
  6. Delegate to AnalystAgent for synthesis
  7. Attach citations via CitationAgent
  8. Return structured ResearchResponse
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from src.agents.analyst_agent import AnalystAgent, AnalysisResult
from src.agents.citation_agent import CitationAgent
from src.ingestion.embedder import Embedder
from src.retrieval.faiss_store import FAISSStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid import HybridRetriever, build_filters
from src.retrieval.reranker import RankedResult, Reranker
from src.utils.cache import CacheClient, cached, make_cache_key
from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.utils.metrics import track_latency

log = get_logger(__name__)


@dataclass
class QueryFilters:
    ticker: str | None = None
    company: str | None = None
    filing_type: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    doc_type: str | None = None


@dataclass
class ResearchResponse:
    question: str
    answer: str
    citations: list[dict[str, Any]]
    retrieved_chunks: int
    filters_applied: dict[str, Any]
    latency_ms: float
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ResearchAgent:
    """
    Orchestrates the full RAG query pipeline.

    Parameters
    ----------
    faiss_store : FAISSStore
    bm25_retriever : BM25Retriever
    embedder : Embedder | None
    cache : CacheClient | None
    """

    def __init__(
        self,
        faiss_store: FAISSStore,
        bm25_retriever: BM25Retriever,
        embedder: Embedder | None = None,
        cache: CacheClient | None = None,
    ) -> None:
        cfg = get_settings()
        self.settings = cfg
        self.embedder = embedder or Embedder()
        self.retriever = HybridRetriever(faiss_store, bm25_retriever)
        self.reranker = Reranker()
        self.analyst = AnalystAgent()
        self.citation_agent = CitationAgent()
        self.cache = cache  # CacheClient or None

    async def query(
        self,
        question: str,
        filters: QueryFilters | None = None,
        top_k: int | None = None,
        stream: bool = False,
    ) -> ResearchResponse:
        """
        Answer a financial research question using hybrid RAG.

        Parameters
        ----------
        question : str
        filters : QueryFilters | None
            Optional metadata constraints.
        top_k : int | None
            Override number of retrieved chunks.
        stream : bool
            If True, use streaming LLM (not yet wired to HTTP layer here).

        Returns
        -------
        ResearchResponse
        """
        import time
        t0 = time.perf_counter()

        # Cache check
        cache_key = make_cache_key(
            "query",
            question=question,
            filters=filters.__dict__ if filters else {},
            top_k=top_k,
        )
        if self.cache:
            cached_val = await self.cache.get(cache_key)
            if cached_val:
                log.info("query_cache_hit", question=question[:80])
                return ResearchResponse(**cached_val)

        log.info("research_query_start", question=question[:120])

        with track_latency("full_query_pipeline"):
            # 1. Build metadata filters
            meta_filters = build_filters(
                ticker=filters.ticker if filters else None,
                company=filters.company if filters else None,
                filing_type=filters.filing_type if filters else None,
                doc_type=filters.doc_type if filters else None,
            ) if filters else None

            # 2. Embed query
            with track_latency("embed_query"):
                query_vector = await self.embedder.embed_query(question)

            # 3. Hybrid retrieval
            hybrid_results = await asyncio.to_thread(
                self.retriever.retrieve,
                query=question,
                query_vector=query_vector,
                top_k=(top_k or self.settings.top_k_semantic) * 2,
                filters=meta_filters,
            )

            if not hybrid_results:
                log.warning("no_results_retrieved", question=question[:80])
                return ResearchResponse(
                    question=question,
                    answer="No relevant documents found for your query. Please try broader search terms or check your filters.",
                    citations=[],
                    retrieved_chunks=0,
                    filters_applied=meta_filters or {},
                    latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                    model=self.settings.openai_model,
                )

            # 4. Rerank
            ranked: list[RankedResult] = await asyncio.to_thread(
                self.reranker.rerank,
                query=question,
                candidates=hybrid_results,
                top_k=top_k or self.settings.top_k_reranked,
            )

            # 5. Date-range post-filter (if requested)
            if filters and (filters.date_from or filters.date_to):
                ranked = self._apply_date_filter(
                    ranked, filters.date_from, filters.date_to
                )

            # 6. Synthesise with AnalystAgent
            analysis: AnalysisResult = await self.analyst.analyse(
                question=question,
                chunks=ranked,
            )

            # 7. Attach citations
            citations = self.citation_agent.build_citations(ranked)

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        response = ResearchResponse(
            question=question,
            answer=analysis.answer,
            citations=citations,
            retrieved_chunks=len(ranked),
            filters_applied=meta_filters or {},
            latency_ms=latency_ms,
            model=self.settings.openai_model,
            metadata={
                "hybrid_results": len(hybrid_results),
                "reranked": len(ranked),
                "intent": analysis.intent,
            },
        )

        if self.cache:
            await self.cache.set(cache_key, response.__dict__)

        log.info(
            "research_query_done",
            question=question[:80],
            latency_ms=latency_ms,
            chunks=len(ranked),
        )
        return response

    def _apply_date_filter(
        self,
        results: list[RankedResult],
        date_from: str | None,
        date_to: str | None,
    ) -> list[RankedResult]:
        """Post-filter by filed_date / period in metadata."""
        filtered = []
        for r in results:
            filed = r.metadata.get("filed_date") or r.metadata.get("period", "")
            if date_from and filed < date_from:
                continue
            if date_to and filed > date_to:
                continue
            filtered.append(r)
        return filtered
