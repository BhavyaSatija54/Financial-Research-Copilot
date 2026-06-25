"""
End-to-end ingestion pipeline.

Stages:
  1. Load raw documents (from disk or SEC EDGAR)
  2. Chunk documents into token-bounded segments
  3. Embed chunks via OpenAI
  4. Upsert vectors into FAISS
  5. Upsert tokens into BM25 index
  6. Persist metadata store

Entry point: IngestionPipeline.run()
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.ingestion.chunker import DocumentChunker, TextChunk
from src.ingestion.embedder import Embedder
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.faiss_store import FAISSStore
from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.utils.metrics import track_latency

log = get_logger(__name__)


@dataclass
class IngestionResult:
    doc_id: str
    chunks_added: int
    tokens_indexed: int
    status: str  # "success" | "error"
    error: str | None = None


@dataclass
class PipelineStats:
    total_documents: int = 0
    total_chunks: int = 0
    total_tokens: int = 0
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


class IngestionPipeline:
    """
    Orchestrates the full document → vector-index pipeline.

    Parameters
    ----------
    faiss_store : FAISSStore
        Pre-initialised vector store.
    bm25_retriever : BM25Retriever
        Pre-initialised BM25 index.
    embedder : Embedder | None
        If None, a new Embedder is created.
    chunker : DocumentChunker | None
        If None, a new DocumentChunker is created.
    """

    def __init__(
        self,
        faiss_store: FAISSStore,
        bm25_retriever: BM25Retriever,
        embedder: Embedder | None = None,
        chunker: DocumentChunker | None = None,
    ) -> None:
        self.faiss = faiss_store
        self.bm25 = bm25_retriever
        self.embedder = embedder or Embedder()
        self.chunker = chunker or DocumentChunker()
        self.settings = get_settings()

    async def ingest_file(
        self,
        path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """Ingest a single file from disk."""
        doc_id = path.stem
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            meta = {"source_file": str(path), "doc_type": "file", **(metadata or {})}
            return await self._process_document(doc_id, text, meta)
        except Exception as exc:
            log.error("ingest_file_error", path=str(path), error=str(exc))
            return IngestionResult(doc_id=doc_id, chunks_added=0, tokens_indexed=0,
                                   status="error", error=str(exc))

    async def ingest_text(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """Ingest raw text with an explicit doc_id."""
        return await self._process_document(doc_id, text, metadata or {})

    async def ingest_directory(
        self,
        directory: Path,
        glob: str = "**/*.txt",
        metadata_fn: Any = None,
    ) -> PipelineStats:
        """
        Ingest all matching files in a directory.

        Parameters
        ----------
        directory : Path
        glob : str
            Glob pattern relative to directory.
        metadata_fn : callable | None
            Optional function(path) -> dict to attach per-file metadata.
        """
        files = sorted(directory.glob(glob))
        log.info("ingest_directory_start", directory=str(directory), files=len(files))
        stats = PipelineStats()

        tasks = [
            self.ingest_file(f, metadata_fn(f) if metadata_fn else None)
            for f in files
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        for result in results:
            if result.status == "success":
                stats.total_documents += 1
                stats.total_chunks += result.chunks_added
                stats.total_tokens += result.tokens_indexed
            else:
                stats.errors.append(f"{result.doc_id}: {result.error}")

        # Save both indexes
        await asyncio.gather(
            asyncio.to_thread(self.faiss.save),
            asyncio.to_thread(self.bm25.save),
        )

        log.info(
            "ingest_directory_done",
            documents=stats.total_documents,
            chunks=stats.total_chunks,
            tokens=stats.total_tokens,
            errors=len(stats.errors),
        )
        return stats

    # ── Private helpers ───────────────────────────────────────────────

    async def _process_document(
        self,
        doc_id: str,
        text: str,
        metadata: dict,
    ) -> IngestionResult:
        """Core pipeline: chunk → embed → index."""
        with track_latency("ingest_document", doc_id=doc_id):
            # 1. Chunk
            chunks: list[TextChunk] = await asyncio.to_thread(
                self.chunker.chunk_document, text, doc_id, metadata
            )

            if not chunks:
                log.warning("no_chunks_produced", doc_id=doc_id)
                return IngestionResult(doc_id=doc_id, chunks_added=0, tokens_indexed=0,
                                       status="error", error="no chunks produced")

            # 2. Embed
            with track_latency("embed_chunks", doc_id=doc_id, n_chunks=len(chunks)):
                embeddings = await self.embedder.embed_chunks(chunks)

            # 3. Index into FAISS
            await asyncio.to_thread(self.faiss.add_chunks, chunks, embeddings)

            # 4. Index into BM25
            await asyncio.to_thread(self.bm25.add_chunks, chunks)

            # 5. Persist metadata
            await asyncio.to_thread(self._append_metadata, chunks)

            total_tokens = sum(c.token_count for c in chunks)
            log.info(
                "document_ingested",
                doc_id=doc_id,
                chunks=len(chunks),
                tokens=total_tokens,
            )
            return IngestionResult(
                doc_id=doc_id,
                chunks_added=len(chunks),
                tokens_indexed=total_tokens,
                status="success",
            )

    def _append_metadata(self, chunks: list[TextChunk]) -> None:
        """Append chunk metadata to JSONL store."""
        path = self.settings.metadata_store_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict()) + "\n")
