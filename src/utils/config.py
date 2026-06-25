"""
Centralised application settings using pydantic-settings.

All values can be overridden via environment variables or a .env file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Environment ──────────────────────────────────────────────────
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    # ── OpenAI / LLM ─────────────────────────────────────────────────
    openai_api_key: SecretStr = Field(..., description="OpenAI API key")
    openai_model: str = "gpt-4o"
    openai_temperature: float = 0.1
    openai_max_tokens: int = 4096

    # ── Embeddings ───────────────────────────────────────────────────
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    embedding_batch_size: int = 512

    # ── Vector / BM25 Store ──────────────────────────────────────────
    faiss_index_path: Path = Path("data/processed/faiss_index")
    bm25_index_path: Path = Path("data/processed/bm25_index.pkl")
    metadata_store_path: Path = Path("data/processed/metadata.jsonl")

    # ── Chunking ─────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_length: int = 100

    # ── Retrieval ────────────────────────────────────────────────────
    top_k_semantic: int = 20
    top_k_bm25: int = 20
    top_k_reranked: int = 8
    rrf_k: int = 60
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── API ──────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_reload: bool = False
    api_key: SecretStr = Field(default="change-me", description="Static API key for auth")

    # ── JWT ──────────────────────────────────────────────────────────
    jwt_secret_key: SecretStr = Field(default="change-me", description="JWT signing secret")
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60

    # ── Rate Limiting ────────────────────────────────────────────────
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 500

    # ── Redis ────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600
    cache_enabled: bool = True

    # ── SEC EDGAR ────────────────────────────────────────────────────
    sec_user_agent: str = "FinancialResearchCopilot/1.0 user@example.com"
    sec_request_delay_seconds: float = 0.1

    # ── Logging ──────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "text"] = "json"
    log_file: Path | None = None

    # ── Derived helpers ──────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def openai_api_key_str(self) -> str:
        return self.openai_api_key.get_secret_value()

    @property
    def api_key_str(self) -> str:
        return self.api_key.get_secret_value()

    @field_validator("faiss_index_path", "bm25_index_path", "metadata_store_path", mode="before")
    @classmethod
    def _ensure_parent_dirs(cls, v: str | Path) -> Path:
        p = Path(v)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton Settings instance."""
    return Settings()  # type: ignore[call-arg]
