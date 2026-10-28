"""Offline tests for the shared TTL cache (ICache) and backend selection."""

import sys
from types import SimpleNamespace

import pytest

from src.tools.cache import MemoryCache, RedisCache, build_cache
from src.tools.settings import settings


async def test_memory_roundtrip_and_expiry(monkeypatch):
    import time

    cache = MemoryCache()
    await cache.set("k", {"a": 1}, ttl_seconds=60.0)
    assert await cache.get("k") == {"a": 1}
    assert await cache.ttl("k") > 0.0

    real_monotonic = time.monotonic
    monkeypatch.setattr(
        "src.tools.cache.time.monotonic", lambda: real_monotonic() + 61
    )
    assert await cache.get("k") is None
    assert await cache.ttl("k") == 0.0


async def test_incr_anchors_window_and_counts():
    cache = MemoryCache()
    assert await cache.incr("c", ttl_seconds=60.0) == 1
    assert await cache.incr("c", ttl_seconds=60.0) == 2
    assert await cache.incr("other", ttl_seconds=60.0) == 1
    ttl = await cache.ttl("c")
    assert 0.0 < ttl <= 60.0


async def test_delete_removes_key():
    cache = MemoryCache()
    await cache.set("k", "v", ttl_seconds=60.0)
    await cache.delete("k")
    assert await cache.get("k") is None
    assert await cache.ttl("k") == 0.0


def test_build_cache_defaults_to_memory(monkeypatch):
    monkeypatch.setattr(settings, "REDIS_URL", "")
    assert isinstance(build_cache(), MemoryCache)


def test_build_cache_degrades_without_redis_package(monkeypatch):
    monkeypatch.setattr(settings, "REDIS_URL", "redis://localhost:6379/0")
    # None in sys.modules makes `import redis` raise ImportError.
    monkeypatch.setitem(sys.modules, "redis", None)
    assert isinstance(build_cache(), MemoryCache)


@pytest.fixture
def failing_redis(monkeypatch):
    """A stub `redis` package whose client always fails."""

    class FailingClient:
        async def get(self, key):
            raise OSError("down")

        async def set(self, *args, **kwargs):
            raise OSError("down")

        async def delete(self, *args, **kwargs):
            raise OSError("down")

        async def pttl(self, key):
            raise OSError("down")

        def pipeline(self, transaction=False):
            raise OSError("down")

    fake_asyncio = SimpleNamespace(
        Redis=SimpleNamespace(from_url=lambda url, **kwargs: FailingClient())
    )
    fake_redis = SimpleNamespace(asyncio=fake_asyncio)
    monkeypatch.setitem(sys.modules, "redis", fake_redis)
    monkeypatch.setitem(sys.modules, "redis.asyncio", fake_asyncio)


async def test_redis_cache_degrades_to_memory(failing_redis, monkeypatch):
    monkeypatch.setattr(settings, "REDIS_URL", "redis://localhost:6379/0")
    cache = build_cache()
    assert isinstance(cache, RedisCache)

    await cache.set("k", "v", ttl_seconds=60.0)
    assert await cache.get("k") == "v"
    assert await cache.incr("c", ttl_seconds=60.0) == 1
    assert await cache.ttl("c") > 0.0


async def test_rate_limiter_survives_redis_outage(failing_redis, monkeypatch):
    from src.modules.api_keys.rate_limit import RateLimiter

    monkeypatch.setattr(settings, "REDIS_URL", "redis://localhost:6379/0")
    limiter = RateLimiter(limit_per_minute=1, cache=build_cache())
    await limiter.check("k")
    with pytest.raises(Exception, match="rate limit"):
        await limiter.check("k")
