"""
POST /ingest  — document ingestion endpoints.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from src.api.dependencies import get_ingestion_pipeline, verify_api_key
from src.api.schemas import (
    ErrorResponse,
    IngestResponse,
    IngestSECRequest,
    IngestTextRequest,
)
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.sec_fetcher import SECFetcher
from src.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post(
    "/text",
    response_model=IngestResponse,
    summary="Ingest raw text document",
)
async def ingest_text(
    body: IngestTextRequest,
    pipeline: Annotated[IngestionPipeline, Depends(get_ingestion_pipeline)],
    _key: Annotated[str, Depends(verify_api_key)],
) -> IngestResponse:
    """Ingest a single plain-text document with metadata."""
    log.info("ingest_text_request", doc_id=body.doc_id, text_len=len(body.text))
    result = await pipeline.ingest_text(
        doc_id=body.doc_id,
        text=body.text,
        metadata=body.metadata,
    )
    return IngestResponse(
        status=result.status,
        documents_processed=1 if result.status == "success" else 0,
        chunks_added=result.chunks_added,
        tokens_indexed=result.tokens_indexed,
        errors=[result.error] if result.error else [],
    )


@router.post(
    "/file",
    response_model=IngestResponse,
    summary="Ingest uploaded file (txt, pdf text layer)",
)
async def ingest_file(
    file: UploadFile,
    pipeline: Annotated[IngestionPipeline, Depends(get_ingestion_pipeline)],
    _key: Annotated[str, Depends(verify_api_key)],
) -> IngestResponse:
    """Upload and ingest a text file directly."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    content = await file.read()
    try:
        text = content.decode("utf-8", errors="replace")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode file: {exc}") from exc

    doc_id = file.filename.rsplit(".", 1)[0]
    log.info("ingest_file_request", filename=file.filename, bytes=len(content))

    result = await pipeline.ingest_text(
        doc_id=doc_id,
        text=text,
        metadata={"source_file": file.filename, "doc_type": "upload"},
    )
    return IngestResponse(
        status=result.status,
        documents_processed=1 if result.status == "success" else 0,
        chunks_added=result.chunks_added,
        tokens_indexed=result.tokens_indexed,
        errors=[result.error] if result.error else [],
    )


@router.post(
    "/sec",
    response_model=IngestResponse,
    summary="Fetch and ingest SEC EDGAR filings",
)
async def ingest_sec(
    body: IngestSECRequest,
    pipeline: Annotated[IngestionPipeline, Depends(get_ingestion_pipeline)],
    _key: Annotated[str, Depends(verify_api_key)],
) -> IngestResponse:
    """Fetch SEC EDGAR filings for a ticker and ingest them into the index."""
    log.info(
        "ingest_sec_request",
        ticker=body.ticker,
        filing_type=body.filing_type,
        limit=body.limit,
    )

    total_chunks = 0
    total_tokens = 0
    errors: list[str] = []
    docs_processed = 0

    async with SECFetcher() as fetcher:
        try:
            filings = await fetcher.get_filings(
                ticker=body.ticker,
                filing_type=body.filing_type,  # type: ignore[arg-type]
                limit=body.limit,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"SEC EDGAR fetch failed: {exc}",
            ) from exc

        tasks = [
            pipeline.ingest_text(
                doc_id=filing.doc_id,
                text=filing.text,
                metadata=filing.metadata,
            )
            for filing in filings
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        for result in results:
            if result.status == "success":
                docs_processed += 1
                total_chunks += result.chunks_added
                total_tokens += result.tokens_indexed
            else:
                errors.append(result.error or "unknown error")

    return IngestResponse(
        status="success" if not errors else "partial",
        documents_processed=docs_processed,
        chunks_added=total_chunks,
        tokens_indexed=total_tokens,
        errors=errors,
    )
