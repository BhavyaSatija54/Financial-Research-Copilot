"""Ingestion endpoints: /api/v1/ingest/sec, /macro, /text, /file, /documents."""
from __future__ import annotations
import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from src.api.dependencies import get_ingestion_pipeline, get_metadata_db, verify_api_key
from src.api.schemas import (
    DocumentListResponse, ErrorResponse, IngestMacroRequest,
    IngestResponse, IngestSECRequest, IngestTextRequest,
)
from src.ingestion.fred_fetcher import MACRO_SERIES, FREDFetcher
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.sec_fetcher import SECFetcher
from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.utils.metadata_db import MetadataDB

log = get_logger(__name__)
router = APIRouter(prefix="/api/v1/ingest", tags=["Ingestion"])


@router.post("/text", response_model=IngestResponse, summary="Ingest raw text document")
async def ingest_text(
    body: IngestTextRequest,
    pipeline: Annotated[IngestionPipeline, Depends(get_ingestion_pipeline)],
    _key: Annotated[str, Depends(verify_api_key)],
) -> IngestResponse:
    result = await pipeline.ingest_text(doc_id=body.doc_id, text=body.text, metadata=body.metadata)
    return IngestResponse(
        status=result.status,
        documents_processed=1 if result.status == "success" else 0,
        chunks_added=result.chunks_added,
        tokens_indexed=result.tokens_indexed,
        errors=[result.error] if result.error else [],
    )


@router.post("/file", response_model=IngestResponse, summary="Ingest uploaded file")
async def ingest_file(
    file: UploadFile,
    pipeline: Annotated[IngestionPipeline, Depends(get_ingestion_pipeline)],
    _key: Annotated[str, Depends(verify_api_key)],
) -> IngestResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    content = await file.read()
    text = content.decode("utf-8", errors="replace")
    doc_id = file.filename.rsplit(".", 1)[0]
    result = await pipeline.ingest_text(
        doc_id=doc_id, text=text,
        metadata={"source_file": file.filename, "doc_type": "upload"},
    )
    return IngestResponse(
        status=result.status,
        documents_processed=1 if result.status == "success" else 0,
        chunks_added=result.chunks_added,
        tokens_indexed=result.tokens_indexed,
        errors=[result.error] if result.error else [],
    )


@router.post("/sec", response_model=IngestResponse, summary="Fetch & ingest SEC EDGAR filings")
async def ingest_sec(
    body: IngestSECRequest,
    pipeline: Annotated[IngestionPipeline, Depends(get_ingestion_pipeline)],
    _key: Annotated[str, Depends(verify_api_key)],
) -> IngestResponse:
    log.info("ingest_sec_request", ticker=body.ticker, filing_type=body.filing_type)
    total_chunks = total_tokens = docs = 0
    errors: list[str] = []

    async with SECFetcher() as fetcher:
        try:
            filings = await fetcher.get_filings(
                ticker=body.ticker, filing_type=body.filing_type, limit=body.limit  # type: ignore
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"SEC EDGAR fetch failed: {exc}") from exc

        results = await asyncio.gather(
            *[pipeline.ingest_text(doc_id=f.doc_id, text=f.text, metadata=f.metadata)
              for f in filings],
            return_exceptions=False,
        )

    for r in results:
        if r.status == "success":
            docs += 1; total_chunks += r.chunks_added; total_tokens += r.tokens_indexed
        else:
            errors.append(r.error or "unknown")

    return IngestResponse(
        status="success" if not errors else "partial",
        documents_processed=docs,
        chunks_added=total_chunks,
        tokens_indexed=total_tokens,
        errors=errors,
    )


@router.post("/macro", response_model=IngestResponse, summary="Fetch & ingest FRED macro data")
async def ingest_macro(
    body: IngestMacroRequest,
    pipeline: Annotated[IngestionPipeline, Depends(get_ingestion_pipeline)],
    _key: Annotated[str, Depends(verify_api_key)],
) -> IngestResponse:
    settings = get_settings()
    api_key = settings.fred_api_key_str
    log.info("ingest_macro_request", series=body.series_ids, has_key=bool(api_key))

    # Build series list
    if body.series_ids:
        series_map = {sid: (name, cat) for sid, name, cat in MACRO_SERIES}
        targets = [
            (sid, series_map.get(sid, (sid, "macro"))[0], series_map.get(sid, (sid, "macro"))[1])
            for sid in body.series_ids
        ]
    else:
        targets = None  # all defaults

    async with FREDFetcher(api_key=api_key) as fetcher:
        docs = await fetcher.fetch_all(series=targets, observation_start=body.observation_start)

    total_chunks = total_tokens = 0
    errors: list[str] = []

    results = await asyncio.gather(
        *[pipeline.ingest_text(doc_id=d.doc_id, text=d.text, metadata=d.metadata) for d in docs],
        return_exceptions=False,
    )
    for r in results:
        if r.status == "success":
            total_chunks += r.chunks_added; total_tokens += r.tokens_indexed
        else:
            errors.append(r.error or "unknown")

    return IngestResponse(
        status="success" if not errors else "partial",
        documents_processed=len(docs) - len(errors),
        chunks_added=total_chunks,
        tokens_indexed=total_tokens,
        errors=errors,
    )


@router.get("/documents", response_model=DocumentListResponse, summary="List indexed documents")
async def list_documents(
    ticker: str | None = None,
    filing_type: str | None = None,
    doc_type: str | None = None,
    limit: int = 50,
    db: MetadataDB = Depends(get_metadata_db),
    _key: str = Depends(verify_api_key),
) -> DocumentListResponse:
    docs = db.list_documents(ticker=ticker, filing_type=filing_type, doc_type=doc_type, limit=limit)
    return DocumentListResponse(documents=docs, total=len(docs))
