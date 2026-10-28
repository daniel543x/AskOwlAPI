"""ICache: small async TTL cache shared across modules (decision 17).

In-process memory is the default backend; setting REDIS_URL selects Redis
(the `redis` package is an optional extra, imported lazily). Consumers cache
small TTL state — rate-limit windows today, model capability probes from
PR 9 later. Never LLM answers; heavy page blobs stay in Postgres
(scraped_page). Any Redis failure permanently degrades the instance to its
in-process fallback: a cache must never take the API down.
"""

import json
import logging
import time
from typing import Any, Protocol

from .settings import settings

logger = logging.getLogger(__name__)


class ICache(Protocol):
    """Minimal async TTL-cache surface; every backend implements exactly this."""

    async def get(self, key: str) -> Any | None: ...

    async def set(self, key: str, value: Any, ttl_seconds: float) -> None: ...

    async def delete(self, key: str) -> None: ...

    async def incr(self, key: str, ttl_seconds: float) -> int:
        """Increment a counter, (re)anchoring its TTL on the first increment."""
        ...

    async def ttl(self, key: str) -> float:
        """Seconds until `key` expires; 0.0 when absent or already expired."""
        ...


class MemoryCache:
    """Single-process TTL cache (the default backend)."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[float, Any]] = {}

    async def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.monotonic() >= expires_at:
            del self._store[key]
            return None
        return value

    async def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        self._store[key] = (time.monotonic() + ttl_seconds, value)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def incr(self, key: str, ttl_seconds: float) -> int:
        now = time.monotonic()
        entry = self._store.get(key)
        if entry is None or now >= entry[0]:
            self._store[key] = (now + ttl_seconds, 1)
            return 1
        count = int(entry[1]) + 1
        self._store[key] = (entry[0], count)
        return count

    async def ttl(self, key: str) -> float:
        entry = self._store.get(key)
        if entry is None:
            return 0.0
        return max(0.0, entry[0] - time.monotonic())


class RedisCache:
    """Redis-backed ICache selected by REDIS_URL.

    The `redis` package is imported lazily so it stays an optional dependency.
    The first Redis failure flips the instance to its in-process fallback for
    the remaining process lifetime (degrade, never crash); a restart retries
    Redis. Counters use INCR + EXPIRE NX, so the TTL anchors at the first
    increment exactly like the in-memory backend.
    """

    def __init__(self, url: str) -> None:
        import redis.asyncio as aioredis  # pyright: ignore[reportMissingImports]

        self._client = aioredis.Redis.from_url(url, decode_responses=True)
        self._fallback = MemoryCache()
        self._degraded = False

    def _degrade(self, exc: Exception) -> None:
        self._degraded = True
        logger.warning(
            "Redis cache failed (%s); serving from the in-process cache.", exc
        )

    async def get(self, key: str) -> Any | None:
        if not self._degraded:
            try:
                raw = await self._client.get(key)
                return None if raw is None else json.loads(raw)
            except ValueError:
                logger.warning(
                    "Undecodable cache value for %s; treating as a miss.", key
                )
                return None
            except Exception as exc:
                self._degrade(exc)
        return await self._fallback.get(key)

    async def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        try:
            raw = json.dumps(value)
        except TypeError:
            logger.error(
                "Cache value for %s is not JSON-serializable; not stored.", key
            )
            return
        if not self._degraded:
            try:
                await self._client.set(key, raw, ex=max(1, int(ttl_seconds)))
                return
            except Exception as exc:
                self._degrade(exc)
        await self._fallback.set(key, value, ttl_seconds)

    async def delete(self, key: str) -> None:
        if not self._degraded:
            try:
                await self._client.delete(key)
                return
            except Exception as exc:
                self._degrade(exc)
        await self._fallback.delete(key)

    async def incr(self, key: str, ttl_seconds: float) -> int:
        if not self._degraded:
            try:
                pipe = self._client.pipeline(transaction=False)
                pipe.incr(key)
                pipe.expire(key, max(1, round(ttl_seconds)), nx=True)
                count, _ = await pipe.execute()
                return int(count)
            except Exception as exc:
                self._degrade(exc)
        return await self._fallback.incr(key, ttl_seconds)

    async def ttl(self, key: str) -> float:
        if not self._degraded:
            try:
                milliseconds = await self._client.pttl(key)
                return max(0.0, milliseconds / 1000.0)
            except Exception as exc:
                self._degrade(exc)
        return await self._fallback.ttl(key)


def build_cache() -> ICache:
    """Pick the backend: Redis when REDIS_URL is set, memory otherwise.

    A missing `redis` package falls back to memory with a warning at startup;
    connection failures degrade later, per operation (see RedisCache).
    """
    url = settings.REDIS_URL
    if not url:
        return MemoryCache()
    try:
        return RedisCache(url)
    except ImportError:
        logger.warning(
            "REDIS_URL is set but the 'redis' package is not installed; "
            "using the in-process cache (install with: uv sync --all-extras)."
        )
        return MemoryCache()
