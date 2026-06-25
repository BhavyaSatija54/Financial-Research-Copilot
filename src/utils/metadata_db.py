"""
SQLite-backed metadata store for indexed chunks.

Replaces the flat JSONL store with a queryable relational layer:
  - Full-text search via SQLite FTS5
  - Filter by ticker, filing_type, doc_type, date range
  - Stats queries for the /health endpoint
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from src.utils.config import get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    text         TEXT NOT NULL,
    token_count  INTEGER,
    chunk_index  INTEGER,
    total_chunks INTEGER,
    ticker       TEXT,
    company      TEXT,
    filing_type  TEXT,
    doc_type     TEXT,
    period       TEXT,
    filed_date   TEXT,
    source_url   TEXT,
    source       TEXT,
    extra_meta   TEXT,   -- JSON blob for arbitrary fields
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id     ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_ticker     ON chunks(ticker);
CREATE INDEX IF NOT EXISTS idx_chunks_filed_date ON chunks(filed_date);
CREATE INDEX IF NOT EXISTS idx_chunks_filing_type ON chunks(filing_type);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    doc_id   UNINDEXED,
    text,
    ticker   UNINDEXED,
    filing_type UNINDEXED,
    content='chunks',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, chunk_id, doc_id, text, ticker, filing_type)
    VALUES (new.rowid, new.chunk_id, new.doc_id, new.text, new.ticker, new.filing_type);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, doc_id, text, ticker, filing_type)
    VALUES ('delete', old.rowid, old.chunk_id, old.doc_id, old.text, old.ticker, old.filing_type);
END;
"""


class MetadataDB:
    """Thread-safe SQLite metadata store."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = str(path or get_settings().metadata_db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
        log.info("metadata_db_ready", path=self._path)

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Write ─────────────────────────────────────────────────────────

    def upsert_chunk(self, chunk_dict: dict[str, Any]) -> None:
        meta = {k: v for k, v in chunk_dict.items()
                if k not in ("chunk_id", "doc_id", "text", "token_count",
                             "chunk_index", "total_chunks", "ticker", "company",
                             "filing_type", "doc_type", "period", "filed_date",
                             "source_url", "source")}
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO chunks
                  (chunk_id, doc_id, text, token_count, chunk_index, total_chunks,
                   ticker, company, filing_type, doc_type, period, filed_date,
                   source_url, source, extra_meta)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    chunk_dict.get("chunk_id"),
                    chunk_dict.get("doc_id"),
                    chunk_dict.get("text", ""),
                    chunk_dict.get("token_count"),
                    chunk_dict.get("chunk_index"),
                    chunk_dict.get("total_chunks"),
                    chunk_dict.get("ticker"),
                    chunk_dict.get("company"),
                    chunk_dict.get("filing_type"),
                    chunk_dict.get("doc_type"),
                    chunk_dict.get("period"),
                    chunk_dict.get("filed_date"),
                    chunk_dict.get("source_url"),
                    chunk_dict.get("source"),
                    json.dumps(meta) if meta else None,
                ),
            )

    def upsert_chunks(self, chunk_dicts: list[dict[str, Any]]) -> None:
        for c in chunk_dicts:
            self.upsert_chunk(c)

    def delete_doc(self, doc_id: str) -> int:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            return cur.rowcount

    # ── Read ──────────────────────────────────────────────────────────

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
        return dict(row) if row else None

    def search_fts(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT c.* FROM chunks c
                JOIN chunks_fts f ON c.chunk_id = f.chunk_id
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_documents(
        self,
        ticker: str | None = None,
        filing_type: str | None = None,
        doc_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if ticker:
            clauses.append("ticker = ?")
            params.append(ticker.upper())
        if filing_type:
            clauses.append("filing_type = ?")
            params.append(filing_type)
        if doc_type:
            clauses.append("doc_type = ?")
            params.append(doc_type)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT doc_id, ticker, company, filing_type, doc_type,
                       period, filed_date, COUNT(*) as chunk_count
                FROM chunks {where}
                GROUP BY doc_id
                ORDER BY filed_date DESC
                LIMIT ?
                """,
                [*params, limit],
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Stats ─────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            docs = conn.execute("SELECT COUNT(DISTINCT doc_id) FROM chunks").fetchone()[0]
            tickers = conn.execute("SELECT COUNT(DISTINCT ticker) FROM chunks WHERE ticker IS NOT NULL").fetchone()[0]
            types = conn.execute(
                "SELECT filing_type, COUNT(*) as n FROM chunks WHERE filing_type IS NOT NULL GROUP BY filing_type"
            ).fetchall()
        return {
            "total_chunks": total,
            "total_documents": docs,
            "total_tickers": tickers,
            "by_filing_type": {r["filing_type"]: r["n"] for r in types},
        }
