"""POST /api/v1/query and POST /api/v1/query/stream endpoints."""
from __future__ import annotations
import asyncio
import json
import uuid
from typing import Annotated, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from src.agents.research_agent import QueryFilters, ResearchAgent, ResearchResponse
from src.api.dependencies import get_research_agent, verify_api_key
from src.api.schemas import ErrorResponse, QueryRequest, QueryResponse
from src.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["Query"])


def _build_filters(body: QueryRequest) -> QueryFilters | None:
    if not body.filters:
        return None
    f = body.filters
    return QueryFilters(
        ticker=f.ticker, company=f.company, filing_type=f.filing_type,
        date_from=f.date_from, date_to=f.date_to, doc_type=f.doc_type,
    )


@router.post(
    "/api/v1/query",
    response_model=QueryResponse,
    responses={401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Financial RAG query",
    description="Submit a natural-language research question. Returns cited answer synthesised from indexed filings, transcripts, and macro data.",
)
async def query(
    request: Request,
    body: QueryRequest,
    agent: Annotated[ResearchAgent, Depends(get_research_agent)],
    _key: Annotated[str, Depends(verify_api_key)],
) -> QueryResponse:
    request_id = str(uuid.uuid4())[:8]
    question = body.get_query()
    log.info("query_request", request_id=request_id, question=question[:100])
    try:
        response: ResearchResponse = await agent.query(
            question=question, filters=_build_filters(body), top_k=body.top_k,
        )
        return QueryResponse(
            query=response.question,
            answer=response.answer,
            citations=response.citations,
            retrieved_chunks=response.retrieved_chunks,
            filters_applied=response.filters_applied,
            latency_ms=response.latency_ms,
            model=response.model,
            metadata=response.metadata,
        )
    except Exception as exc:
        log.error("query_error", request_id=request_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post(
    "/api/v1/query/stream",
    summary="Streaming RAG query (SSE)",
    description="Same as /query but streams tokens via Server-Sent Events.",
)
async def query_stream(
    request: Request,
    body: QueryRequest,
    agent: Annotated[ResearchAgent, Depends(get_research_agent)],
    _key: Annotated[str, Depends(verify_api_key)],
) -> StreamingResponse:
    question = body.get_query()

    async def _generate() -> AsyncIterator[str]:
        try:
            yield f"data: {json.dumps({'type': 'start', 'question': question})}\n\n"
            response = await agent.query(question=question, filters=_build_filters(body), top_k=body.top_k)
            words = response.answer.split()
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield f"data: {json.dumps({'type': 'token', 'token': chunk})}\n\n"
                await asyncio.sleep(0.012)
            final_payload = {
                "type": "done",
                "citations": response.citations,
                "retrieved_chunks": response.retrieved_chunks,
                "latency_ms": response.latency_ms,
            }
            yield f"data: {json.dumps(final_payload)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
