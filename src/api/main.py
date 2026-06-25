"""FastAPI application factory with full middleware stack."""
from __future__ import annotations
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from src.api.dependencies import initialise_services, shutdown_services
from src.api.routes import health, ingest, query
from src.utils.config import get_settings
from src.utils.logger import configure_logging, get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(
        level=settings.log_level,
        fmt=settings.log_format,
        log_file=str(settings.log_file) if settings.log_file else None,
    )
    log.info("app_startup", environment=settings.environment)
    await initialise_services(settings)
    yield
    log.info("app_shutdown")
    await shutdown_services()


def create_app() -> FastAPI:
    settings = get_settings()
    limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

    app = FastAPI(
        title="Financial Research Copilot",
        description=(
            "Hybrid RAG research assistant for SEC filings (10-K/10-Q/8-K), "
            "earnings transcripts, and FRED macroeconomic data. "
            "Combines FAISS semantic search + BM25 keyword retrieval with "
            "Reciprocal Rank Fusion and GPT-4o synthesis."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else ["http://localhost:8501"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    @app.middleware("http")
    async def request_logger(request: Request, call_next: any) -> Response:
        request_id = str(uuid.uuid4())[:8]
        t0 = time.perf_counter()
        response: Response = await call_next(request)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        log.info("http_request", method=request.method, path=request.url.path,
                 status=response.status_code, latency_ms=ms, request_id=request_id)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-Ms"] = str(ms)
        return response

    @app.exception_handler(Exception)
    async def global_handler(request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled_exception", error=str(exc), path=request.url.path)
        return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": str(exc)})

    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(ingest.router)
    return app


app = create_app()
