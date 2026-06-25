"""
BM25 keyword retriever built on rank-bm25.

Supports:
  - Incremental chunk addition
  - Metadata-filtered search
  - Pickle-based persistence
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.ingestion.chunker import TextChunk
from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.utils.metrics import track_latency

log = get_logger(__name__)

_STOPWORDS = frozenset(
    "a an the and or but in on at to for of with is are was were be been "
    "being have has had do does did will would could should may might".split()
)


def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


@dataclass
class BM25Result:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]
    rank: int = 0


class BM25Retriever:
    """BM25Okapi index over TextChunks."""

    def __init__(self) -> None:
        self._corpus_tokens: list[list[str]] = []
        self._chunks: list[TextChunk] = []
        self._index: BM25Okapi | None = None
        self._dirty: bool = False  # rebuild index on next search

    # ── Building ─────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[TextChunk]) -> None:
        for chunk in chunks:
            self._corpus_tokens.append(_tokenise(chunk.text))
            self._chunks.append(chunk)
        self._dirty = True
        log.debug("bm25_add", n=len(chunks), total=len(self._chunks))

    def _rebuild(self) -> None:
        if not self._corpus_tokens:
            self._index = None
            return
        self._index = BM25Okapi(self._corpus_tokens)
        self._dirty = False
        log.debug("bm25_index_rebuilt", corpus_size=len(self._corpus_tokens))

    # ── Searching ────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[BM25Result]:
        if self._dirty or self._index is None:
            self._rebuild()
        if self._index is None:
            log.warning("bm25_empty_index")
            return []

        with track_latency("bm25_search", top_k=top_k):
            query_tokens = _tokenise(query)
            if not query_tokens:
                return []

            raw_scores = self._index.get_scores(query_tokens)

        results: list[BM25Result] = []
        scored = sorted(enumerate(raw_scores), key=lambda x: x[1], reverse=True)

        for idx, score in scored:
            if len(results) >= top_k:
                break
            if score <= 0:
                break
            chunk = self._chunks[idx]
            if filters and not self._matches_filters(chunk.metadata, filters):
                continue
            results.append(
                BM25Result(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=float(score),
                    metadata=chunk.metadata,
                    rank=len(results) + 1,
                )
            )

        log.debug("bm25_search_done", query_tokens=len(query_tokens), returned=len(results))
        return results

    def _matches_filters(self, metadata: dict, filters: dict) -> bool:
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
        p = Path(path or get_settings().bm25_index_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(
                {
                    "corpus_tokens": self._corpus_tokens,
                    "chunks": self._chunks,
                },
                f,
            )
        log.info("bm25_saved", path=str(p), total=len(self._chunks))

    def load(self, path: Path | None = None) -> "BM25Retriever":
        p = Path(path or get_settings().bm25_index_path)
        if not p.exists():
            log.warning("bm25_index_not_found", path=str(p))
            return self
        with open(p, "rb") as f:
            data = pickle.load(f)
        self._corpus_tokens = data["corpus_tokens"]
        self._chunks = data["chunks"]
        self._dirty = True
        log.info("bm25_loaded", path=str(p), total=len(self._chunks))
        return self

    @classmethod
    def from_disk(cls, path: Path | None = None) -> "BM25Retriever":
        r = cls()
        r.load(path)
        return r

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)
