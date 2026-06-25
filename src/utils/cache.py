"""
Redis-backed async cache layer with graceful fallback.

If Redis is unavailable, all operations silently no-op so the
application continues to function without caching.
"""

from __future__ import annotations

import hashlib
import json
from functools import wraps
from typing import Any, Callable

import redis.asyncio as aioredis

from src.utils.logger import get_logger

log = get_logger(__name__)


class CacheClient:
    """Async Redis cache with JSON serialisation."""

    def __init__(self, url: str, ttl: int = 3600, enabled: bool = True) -> None:
        self._url = url
        self._ttl = ttl
        self._enabled = enabled
        self._client: aioredis.Redis | None = None

    async def connect(self) -> None:
        if not self._enabled:
            return
        try:
            self._client = aioredis.from_url(self._url, decode_responses=True)
            await self._client.ping()
            log.info("cache_connected", url=self._url)
        except Exception as exc:
            log.warning("cache_unavailable", error=str(exc))
            self._client = None

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()

    async def get(self, key: str) -> Any | None:
        if not self._client:
            return None
        try:
            value = await self._client.get(key)
            if value is not None:
                log.debug("cache_hit", key=key)
                return json.loads(value)
        except Exception as exc:
            log.warning("cache_get_error", key=key, error=str(exc))
        return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        if not self._client:
            return
        try:
            await self._client.setex(key, ttl or self._ttl, json.dumps(value))
            log.debug("cache_set", key=key, ttl=ttl or self._ttl)
        except Exception as exc:
            log.warning("cache_set_error", key=key, error=str(exc))

    async def delete(self, key: str) -> None:
        if not self._client:
            return
        try:
            await self._client.delete(key)
        except Exception as exc:
            log.warning("cache_delete_error", key=key, error=str(exc))

    async def flush_prefix(self, prefix: str) -> int:
        """Delete all keys matching a prefix. Returns number deleted."""
        if not self._client:
            return 0
        try:
            keys = await self._client.keys(f"{prefix}*")
            if keys:
                return await self._client.delete(*keys)
        except Exception as exc:
            log.warning("cache_flush_error", prefix=prefix, error=str(exc))
        return 0

    @property
    def available(self) -> bool:
        return self._client is not None


def make_cache_key(prefix: str, **kwargs: Any) -> str:
    """Stable, deterministic cache key from kwargs."""
    payload = json.dumps(kwargs, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{prefix}:{digest}"


def cached(prefix: str, ttl: int | None = None) -> Callable:
    """
    Decorator for async methods with a `self.cache: CacheClient` attribute.

    Usage::

        @cached("query", ttl=600)
        async def run_query(self, question: str, filters: dict) -> dict:
            ...
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            cache: CacheClient | None = getattr(self, "cache", None)
            if cache is None or not cache.available:
                return await fn(self, *args, **kwargs)

            key = make_cache_key(prefix, args=args, kwargs=kwargs)
            cached_value = await cache.get(key)
            if cached_value is not None:
                return cached_value

            result = await fn(self, *args, **kwargs)
            await cache.set(key, result, ttl)
            return result

        return wrapper

    return decorator
