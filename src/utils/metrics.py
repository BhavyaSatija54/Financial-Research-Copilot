"""
Retrieval quality metrics and latency instrumentation.

Provides:
  - precision@k, recall@k, MRR, NDCG
  - Latency context manager
  - In-memory metrics store (swap for Prometheus in prod)
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Sequence

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Retrieval Quality ─────────────────────────────────────────────────


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of top-k retrieved docs that are relevant."""
    top_k = list(retrieved)[:k]
    if not top_k:
        return 0.0
    relevant_set = set(relevant)
    return sum(1 for doc_id in top_k if doc_id in relevant_set) / k


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of relevant docs found in top-k retrieved."""
    top_k = list(retrieved)[:k]
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    return sum(1 for doc_id in top_k if doc_id in relevant_set) / len(relevant_set)


def mean_reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """MRR: reciprocal rank of the first relevant document."""
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain @k."""
    top_k = list(retrieved)[:k]
    relevant_set = set(relevant)

    dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank, doc_id in enumerate(top_k, start=1)
        if doc_id in relevant_set
    )

    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


# ── Latency Tracking ─────────────────────────────────────────────────


@dataclass
class LatencyRecord:
    operation: str
    duration_ms: float
    metadata: dict = field(default_factory=dict)


_latency_store: dict[str, list[float]] = defaultdict(list)


@contextmanager
def track_latency(operation: str, **meta: object) -> Generator[None, None, None]:
    """Context manager that logs and records operation latency."""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        _latency_store[operation].append(duration_ms)
        log.debug(
            "latency_recorded",
            operation=operation,
            duration_ms=round(duration_ms, 2),
            **meta,
        )


def get_latency_stats(operation: str) -> dict[str, float]:
    """Return p50/p95/p99/mean for a given operation."""
    samples = _latency_store.get(operation, [])
    if not samples:
        return {}
    arr = np.array(samples)
    return {
        "count": len(samples),
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


def get_all_latency_stats() -> dict[str, dict[str, float]]:
    return {op: get_latency_stats(op) for op in _latency_store}
