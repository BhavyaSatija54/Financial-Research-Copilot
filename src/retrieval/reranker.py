"""
Contextual reranker using a cross-encoder model.

Falls back gracefully to RRF-score ordering when the
cross-encoder model is unavailable (e.g. during tests or
when sentence-transformers is not installed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.retrieval.hybrid import HybridResult
from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.utils.metrics import track_latency

log = get_logger(__name__)


@dataclass
class RankedResult:
    """Final reranked result presented to the agent layer."""

    chunk_id: str
    text: str
    rerank_score: float
    rrf_score: float
    metadata: dict[str, Any]
    rank: int
    sources: list[str]

    @classmethod
    def from_hybrid(cls, h: HybridResult, rerank_score: float, rank: int) -> "RankedResult":
        return cls(
            chunk_id=h.chunk_id,
            text=h.text,
            rerank_score=rerank_score,
            rrf_score=h.rrf_score,
            metadata=h.metadata,
            rank=rank,
            sources=h.sources,
        )


class Reranker:
    """
    Cross-encoder reranker.

    Loads `cross-encoder/ms-marco-MiniLM-L-6-v2` (or whatever is
    configured) at first use and scores (query, passage) pairs.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: Any = None
        self._available: bool = False
        self._tried_load: bool = False

    def _load_model(self) -> None:
        if self._tried_load:
            return
        self._tried_load = True
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._model = CrossEncoder(
                self.settings.reranker_model,
                max_length=512,
            )
            self._available = True
            log.info("reranker_loaded", model=self.settings.reranker_model)
        except Exception as exc:
            log.warning(
                "reranker_unavailable",
                error=str(exc),
                fallback="rrf_score_ordering",
            )

    def rerank(
        self,
        query: str,
        candidates: list[HybridResult],
        top_k: int | None = None,
    ) -> list[RankedResult]:
        """
        Rerank candidates using cross-encoder scores.

        Falls back to RRF score if the model is not available.

        Parameters
        ----------
        query : str
        candidates : list[HybridResult]
        top_k : int | None
            Number of results to return (default: all candidates).

        Returns
        -------
        list[RankedResult]  sorted by descending rerank score
        """
        if not candidates:
            return []

        self._load_model()
        k = top_k or len(candidates)

        with track_latency("rerank", n_candidates=len(candidates)):
            if self._available and self._model is not None:
                scores = self._cross_encoder_scores(query, candidates)
            else:
                scores = [c.rrf_score for c in candidates]

        paired = sorted(
            zip(candidates, scores), key=lambda x: x[1], reverse=True
        )[:k]

        results = [
            RankedResult.from_hybrid(candidate, score, rank)
            for rank, (candidate, score) in enumerate(paired, start=1)
        ]

        log.debug(
            "rerank_done",
            input=len(candidates),
            output=len(results),
            model_used=self._available,
        )
        return results

    def _cross_encoder_scores(
        self, query: str, candidates: list[HybridResult]
    ) -> list[float]:
        pairs = [(query, c.text) for c in candidates]
        raw: list[float] = self._model.predict(pairs, show_progress_bar=False).tolist()
        return raw
