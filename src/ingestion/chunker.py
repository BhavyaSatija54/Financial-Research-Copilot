"""
Contextual, metadata-aware text chunking.

Strategy:
  1. Split on sentence boundaries (respects semantic units).
  2. Accumulate sentences into chunks of target token size.
  3. Apply sliding window overlap between consecutive chunks.
  4. Attach rich metadata to every chunk for downstream filtering.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Iterator

import tiktoken

from src.utils.config import get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)

# Tokenizer shared across all calls
_TOKENIZER: tiktoken.Encoding | None = None


def _get_tokenizer() -> tiktoken.Encoding:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    return _TOKENIZER


def count_tokens(text: str) -> int:
    return len(_get_tokenizer().encode(text))


# ── Data model ───────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """A single processable chunk of a document."""

    chunk_id: str
    text: str
    token_count: int
    char_start: int
    char_end: int
    chunk_index: int
    total_chunks: int  # populated after all chunks are built
    metadata: dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return self.metadata.get("doc_id", "")

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "token_count": self.token_count,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            **self.metadata,
        }


# ── Sentence splitter ─────────────────────────────────────────────────

_SENT_BOUNDARY = re.compile(
    r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+",
    re.MULTILINE,
)

_SECTION_HEADERS = re.compile(
    r"^(ITEM\s+\d+[A-Z]?\.?|PART\s+[IVX]+\.?|SECTION\s+\d+)",
    re.IGNORECASE | re.MULTILINE,
)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving financial section headers."""
    # Insert boundary markers at section headers
    text = _SECTION_HEADERS.sub(r"\n\n\g<0>", text)
    sentences = _SENT_BOUNDARY.split(text)
    return [s.strip() for s in sentences if s.strip()]


# ── Chunker ──────────────────────────────────────────────────────────

class DocumentChunker:
    """
    Sentence-aware sliding-window chunker with token budget.

    Parameters
    ----------
    chunk_size : int
        Target token count per chunk (default from settings).
    chunk_overlap : int
        Overlap tokens between consecutive chunks.
    min_chunk_length : int
        Minimum character length; shorter chunks are discarded.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_length: int | None = None,
    ) -> None:
        cfg = get_settings()
        self.chunk_size = chunk_size or cfg.chunk_size
        self.chunk_overlap = chunk_overlap or cfg.chunk_overlap
        self.min_chunk_length = min_chunk_length or cfg.min_chunk_length

    def chunk_document(
        self,
        text: str,
        doc_id: str,
        metadata: dict | None = None,
    ) -> list[TextChunk]:
        """
        Chunk a full document into overlapping token-bounded segments.

        Parameters
        ----------
        text : str
            Full document text.
        doc_id : str
            Unique document identifier (e.g. AAPL_10-K_2024).
        metadata : dict
            Arbitrary metadata attached to every chunk.

        Returns
        -------
        list[TextChunk]
        """
        if not text.strip():
            log.warning("empty_document_skipped", doc_id=doc_id)
            return []

        sentences = split_sentences(text)
        chunks = list(self._build_chunks(sentences, doc_id, metadata or {}))

        # Back-fill total_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total

        log.info(
            "document_chunked",
            doc_id=doc_id,
            total_chunks=total,
            total_tokens=sum(c.token_count for c in chunks),
        )
        return chunks

    def _build_chunks(
        self,
        sentences: list[str],
        doc_id: str,
        metadata: dict,
    ) -> Iterator[TextChunk]:
        """Sliding-window chunk builder."""
        tokenizer = _get_tokenizer()
        sent_tokens = [tokenizer.encode(s) for s in sentences]

        window: list[int] = []       # token indices
        window_text: list[str] = []  # parallel sentence texts
        char_pos = 0
        chunk_index = 0

        i = 0
        while i < len(sentences):
            s_toks = sent_tokens[i]

            # If a single sentence exceeds chunk_size, force-split it
            if len(s_toks) > self.chunk_size:
                # Flush current window first
                if window:
                    yield self._make_chunk(
                        window_text, window, char_pos, chunk_index, doc_id, metadata
                    )
                    chunk_index += 1
                    char_pos += sum(len(s) + 1 for s in window_text)
                    window, window_text = [], []

                # Hard split the long sentence
                for sub_chunk, sub_text in self._split_long_sentence(
                    sentences[i], s_toks, char_pos, chunk_index, doc_id, metadata
                ):
                    yield sub_chunk
                    chunk_index += 1
                    char_pos += len(sub_text) + 1
                i += 1
                continue

            # Would adding this sentence exceed budget?
            if window and len(window) + len(s_toks) > self.chunk_size:
                yield self._make_chunk(
                    window_text, window, char_pos, chunk_index, doc_id, metadata
                )
                chunk_index += 1
                char_pos += sum(len(s) + 1 for s in window_text)

                # Compute overlap: keep sentences from the tail
                overlap_tokens: list[int] = []
                overlap_texts: list[str] = []
                for s_t, s_txt in zip(reversed(sent_tokens[:i]), reversed(window_text)):
                    if len(overlap_tokens) + len(s_t) > self.chunk_overlap:
                        break
                    overlap_tokens = list(s_t) + overlap_tokens
                    overlap_texts = [s_txt] + overlap_texts

                window = overlap_tokens
                window_text = overlap_texts

            window.extend(s_toks)
            window_text.append(sentences[i])
            i += 1

        # Final chunk
        if window:
            chunk = self._make_chunk(
                window_text, window, char_pos, chunk_index, doc_id, metadata
            )
            if len(chunk.text) >= self.min_chunk_length:
                yield chunk

    def _make_chunk(
        self,
        texts: list[str],
        tokens: list[int],
        char_start: int,
        index: int,
        doc_id: str,
        metadata: dict,
    ) -> TextChunk:
        text = " ".join(texts)
        char_end = char_start + len(text)
        return TextChunk(
            chunk_id=str(uuid.uuid4()),
            text=text,
            token_count=len(tokens),
            char_start=char_start,
            char_end=char_end,
            chunk_index=index,
            total_chunks=0,  # filled in later
            metadata={
                "doc_id": doc_id,
                **metadata,
            },
        )

    def _split_long_sentence(
        self,
        sentence: str,
        tokens: list[int],
        char_start: int,
        start_index: int,
        doc_id: str,
        metadata: dict,
    ) -> Iterator[tuple["TextChunk", str]]:
        """Hard-split a sentence that exceeds chunk_size."""
        tokenizer = _get_tokenizer()
        for i in range(0, len(tokens), self.chunk_size):
            sub_tokens = tokens[i : i + self.chunk_size]
            sub_text = tokenizer.decode(sub_tokens)
            chunk = self._make_chunk(
                [sub_text],
                sub_tokens,
                char_start,
                start_index,
                doc_id,
                metadata,
            )
            yield chunk, sub_text
            char_start += len(sub_text)
            start_index += 1
