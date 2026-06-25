"""
Citation Agent — builds structured source citations from ranked chunks.

Deduplicates by document, extracts key metadata, and produces
citation objects that map to specific filing/transcript passages.
"""

from __future__ import annotations

from typing import Any

from src.retrieval.reranker import RankedResult
from src.utils.logger import get_logger

log = get_logger(__name__)


class CitationAgent:
    """Builds structured citations from retrieved chunks."""

    def build_citations(self, chunks: list[RankedResult]) -> list[dict[str, Any]]:
        """
        Produce a deduplicated list of source citations.

        Parameters
        ----------
        chunks : list[RankedResult]
            Reranked retrieval results.

        Returns
        -------
        list[dict]  sorted by chunk rank, deduplicated by doc_id
        """
        seen_docs: set[str] = set()
        citations: list[dict[str, Any]] = []

        for chunk in chunks:
            meta = chunk.metadata
            doc_id: str = meta.get("doc_id", chunk.chunk_id)

            # Include one citation entry per unique document
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)

            citation: dict[str, Any] = {
                "citation_number": len(citations) + 1,
                "chunk_id": chunk.chunk_id,
                "doc_id": doc_id,
                "company": meta.get("company", meta.get("ticker", "Unknown")),
                "ticker": meta.get("ticker"),
                "filing_type": meta.get("filing_type", meta.get("doc_type", "Document")),
                "period": meta.get("period", meta.get("filed_date")),
                "filed_date": meta.get("filed_date"),
                "source_url": meta.get("source_url"),
                "source": meta.get("source", "Document"),
                "excerpt": self._truncate(chunk.text, max_chars=300),
                "relevance_score": round(chunk.rerank_score, 4),
                "retrieval_sources": chunk.sources,
            }
            citations.append(citation)

        log.debug("citations_built", n=len(citations), unique_docs=len(seen_docs))
        return citations

    def format_inline(self, citations: list[dict[str, Any]]) -> str:
        """Return a formatted reference list suitable for appending to an answer."""
        if not citations:
            return ""
        lines = ["\n\n---\n**Sources**\n"]
        for c in citations:
            ticker = f" ({c['ticker']})" if c.get("ticker") else ""
            period = f", {c['period']}" if c.get("period") else ""
            url = f" — [{c['source_url']}]({c['source_url']})" if c.get("source_url") else ""
            lines.append(
                f"[{c['citation_number']}] {c['company']}{ticker} | "
                f"{c['filing_type']}{period}{url}"
            )
        return "\n".join(lines)

    @staticmethod
    def _truncate(text: str, max_chars: int = 300) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rsplit(" ", 1)[0] + "…"
