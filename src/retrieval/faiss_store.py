"""
FAISS vector store with:
  - Inner-product (cosine) search on L2-normalised vectors
  - Chunk-level metadata linkage
  - Persistence (save / load)
  - Filtered search (pre-filter by metadata before ANN)
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.ingestion.chunker import TextChunk
from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.utils.metrics import track_latency

log = get_logger(__name__)


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float          # cosine similarity [−1, 1]
    metadata: dict[str, Any]
    rank: int = 0


class FAISSStore:
    """
    Flat inner-product FAISS index with metadata side-store.

    For production at scale, swap FlatIP with IVF or HNSW.
    """

    def __init__(self, dimension: int | None = None) -> None:
        cfg = get_settings()
        self._dim = dimension or cfg.embedding_dimension
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(self._dim)
        self._id_map: list[str] = []          # FAISS int id → chunk_id
        self._chunk_map: dict[str, TextChunk] = {}  # chunk_id → TextChunk

    # ── Building ─────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[TextChunk], embeddings: np.ndarray) -> None:
        """Add chunks with their pre-computed embeddings."""
        assert embeddings.ndim == 2 and embeddings.shape[1] == self._dim, (
            f"Expected (n, {self._dim}), got {embeddings.shape}"
        )
        # L2-normalise for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0, 1.0, norms)
        normalised = (embeddings / safe_norms).astype(np.float32)

        self._index.add(normalised)
        for chunk in chunks:
            self._id_map.append(chunk.chunk_id)
            self._chunk_map[chunk.chunk_id] = chunk

        log.debug("faiss_add", n=len(chunks), total=self._index.ntotal)

    # ── Searching ────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Semantic nearest-neighbour search with optional metadata filtering.

        Parameters
        ----------
        query_vector : np.ndarray  shape (dim,)
        top_k : int
        filters : dict | None
            e.g. {"ticker": "AAPL", "filing_type": "10-K"}

        Returns
        -------
        list[SearchResult]  sorted by descending cosine similarity
        """
        if self._index.ntotal == 0:
            log.warning("faiss_empty_index")
            return []

        with track_latency("faiss_search", top_k=top_k):
            # Normalise query
            q = query_vector.astype(np.float32).reshape(1, -1)
            norm = np.linalg.norm(q)
            if norm > 0:
                q = q / norm

            # Over-fetch to allow for post-filtering
            fetch_k = min(top_k * 10 if filters else top_k, self._index.ntotal)
            scores, indices = self._index.search(q, fetch_k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk_id = self._id_map[idx]
            chunk = self._chunk_map.get(chunk_id)
            if chunk is None:
                continue
            if filters and not self._matches_filters(chunk.metadata, filters):
                continue

            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=chunk.text,
                    score=float(score),
                    metadata=chunk.metadata,
                )
            )
            if len(results) >= top_k:
                break

        for rank, r in enumerate(results):
            r.rank = rank + 1

        log.debug("faiss_search_done", returned=len(results))
        return results

    def _matches_filters(self, metadata: dict, filters: dict) -> bool:
        """Return True if chunk metadata matches all filter key/values."""
        for key, value in filters.items():
            meta_val = metadata.get(key)
            if meta_val is None:
                return False
            if isinstance(value, list):
                if meta_val not in value:
                    return False
            elif meta_val != value:
                return False
        return True

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> None:
        p = Path(path or get_settings().faiss_index_path)
        p.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(p / "index.faiss"))
        with open(p / "id_map.pkl", "wb") as f:
            pickle.dump(self._id_map, f)
        with open(p / "chunk_map.pkl", "wb") as f:
            pickle.dump(self._chunk_map, f)

        log.info("faiss_saved", path=str(p), total=self._index.ntotal)

    def load(self, path: Path | None = None) -> "FAISSStore":
        p = Path(path or get_settings().faiss_index_path)

        if not (p / "index.faiss").exists():
            log.warning("faiss_index_not_found", path=str(p))
            return self

        self._index = faiss.read_index(str(p / "index.faiss"))
        with open(p / "id_map.pkl", "rb") as f:
            self._id_map = pickle.load(f)
        with open(p / "chunk_map.pkl", "rb") as f:
            self._chunk_map = pickle.load(f)

        log.info("faiss_loaded", path=str(p), total=self._index.ntotal)
        return self

    @classmethod
    def from_disk(cls, path: Path | None = None) -> "FAISSStore":
        store = cls()
        store.load(path)
        return store

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal
