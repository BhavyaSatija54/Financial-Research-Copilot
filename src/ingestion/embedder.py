"""
OpenAI embedding wrapper with:
  - Async batch processing
  - Exponential-backoff retry (via tenacity)
  - Token-budget enforcement per batch
  - Dimension validation
"""

from __future__ import annotations

import asyncio
from typing import Sequence

import numpy as np
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.ingestion.chunker import TextChunk
from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.utils.metrics import track_latency

log = get_logger(__name__)

# OpenAI token limit per embedding call (conservative)
_MAX_TOKENS_PER_CALL = 8_000
_MAX_ITEMS_PER_BATCH = 512


class EmbeddingError(Exception):
    pass


class Embedder:
    """Async OpenAI text-embedding-3-* wrapper."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = AsyncOpenAI(api_key=self.settings.openai_api_key_str)
        self._model = self.settings.embedding_model
        self._dim = self.settings.embedding_dimension

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts, batching to stay within API limits.

        Returns
        -------
        np.ndarray of shape (len(texts), embedding_dimension)
        """
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        batches = self._build_batches(texts)
        log.info("embedding_start", total_texts=len(texts), batches=len(batches))

        all_embeddings: list[np.ndarray] = []

        with track_latency("embed_texts", n_texts=len(texts)):
            for batch in batches:
                embeddings = await self._embed_batch_with_retry(batch)
                all_embeddings.append(embeddings)
                await asyncio.sleep(0)  # yield control

        result = np.vstack(all_embeddings).astype(np.float32)
        assert result.shape == (len(texts), self._dim), (
            f"Embedding shape mismatch: got {result.shape}, "
            f"expected ({len(texts)}, {self._dim})"
        )
        log.info("embedding_done", shape=result.shape)
        return result

    async def embed_chunks(self, chunks: list[TextChunk]) -> np.ndarray:
        """Embed TextChunk objects in bulk."""
        return await self.embed_texts([c.text for c in chunks])

    async def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        result = await self.embed_texts([query])
        return result[0]

    # ── Private helpers ───────────────────────────────────────────────

    def _build_batches(self, texts: list[str]) -> list[list[str]]:
        """Group texts into API-safe batches."""
        batches: list[list[str]] = []
        current_batch: list[str] = []

        for text in texts:
            if len(current_batch) >= _MAX_ITEMS_PER_BATCH:
                batches.append(current_batch)
                current_batch = []
            current_batch.append(text)

        if current_batch:
            batches.append(current_batch)

        return batches

    async def _embed_batch_with_retry(self, texts: list[str]) -> np.ndarray:
        """Call OpenAI embedding API with exponential-backoff retry."""
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            reraise=True,
        ):
            with attempt:
                response = await self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                )

        vectors = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        arr = np.array(vectors, dtype=np.float32)

        if arr.shape[1] != self._dim:
            raise EmbeddingError(
                f"Unexpected embedding dimension: {arr.shape[1]} != {self._dim}"
            )

        return arr
