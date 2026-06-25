"""
POST /query  — main RAG query endpoint with optional SSE streaming.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Annotated, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.agents.research_agent import QueryFilters, ResearchAgent, ResearchResponse
from src.api.dependencies import get_research_agent, verify_api_key
from src.api.schemas import ErrorResponse, QueryRequest, QueryResponse
from src.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])
limiter = Limiter(key_func=get_remote_address)


@router.post(
    "",
    response_model=QueryResponse,
    responses={
        401: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"description": "Rate limit exceeded"},
        500: {"model": ErrorResponse},
    },
    summary="Run a financial research query",
    description=(
        "Submit a natural-language question. The system retrieves relevant excerpts "
        "from indexed SEC filings, earnings transcripts, and macro reports, then "
        "synthesises a cited answer using GPT-4o."
    ),
)
async def query(
    request: Request,
    body: QueryRequest,
    agent: Annotated[ResearchAgent, Depends(get_research_agent)],
    _key: Annotated[str, Depends(verify_api_key)],
) -> QueryResponse | StreamingResponse:
    request_id = str(uuid.uuid4())[:8]
    log.info(
        "query_request",
        request_id=request_id,
        question=body.question[:100],
        stream=body.stream,
    )

    filters: QueryFilters | None = None
    if body.filters:
        f = body.filters
        filters = QueryFilters(
            ticker=f.ticker,
            company=f.company,
            filing_type=f.filing_type,
            date_from=f.date_from,
            date_to=f.date_to,
            doc_type=f.doc_type,
        )

    try:
        if body.stream:
            return StreamingResponse(
                _stream_response(agent, body.question, filters, body.top_k),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Request-ID": request_id,
                },
            )

        response: ResearchResponse = await agent.query(
            question=body.question,
            filters=filters,
            top_k=body.top_k,
        )
        return QueryResponse(**response.__dict__)

    except Exception as exc:
        log.error("query_error", request_id=request_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {exc}",
        ) from exc


async def _stream_response(
    agent: ResearchAgent,
    question: str,
    filters: QueryFilters | None,
    top_k: int | None,
) -> AsyncIterator[str]:
    """Server-Sent Events generator for streaming responses."""
    try:
        yield "event: start\ndata: {}\n\n"

        response = await agent.query(question=question, filters=filters, top_k=top_k)

        # Stream answer word by word for UX effect
        words = response.answer.split()
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'token': chunk})}\n\n"
            await asyncio.sleep(0.01)

        # Send citations + metadata as final event
        final = {
            "citations": response.citations,
            "retrieved_chunks": response.retrieved_chunks,
            "latency_ms": response.latency_ms,
        }
        yield f"event: done\ndata: {json.dumps(final)}\n\n"

    except Exception as exc:
        log.error("stream_error", error=str(exc))
        yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
