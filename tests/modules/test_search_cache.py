"""Offline tests for the search-result TTL cache (WU-10 / AR-1)."""

import pytest

from src.modules.search_engine.caching import CachedSearchProvider
from src.modules.search_engine.providers.base import ISearchProvider, SearchResult
from src.tools.cache_flags import CACHE_BYPASS, cache_bypass


class CountingProvider(ISearchProvider):
    def __init__(self, results: list[SearchResult] | None = None):
        self.results = (
            [SearchResult("t", "rust s", "https://a.dev/x")]
            if results is None
            else results
        )
        self.calls: list[tuple[str, list[str] | None]] = []

    async def search(self, query, engines=None, categories=None):
        self.calls.append((query, engines))
        return list(self.results)


@pytest.mark.asyncio
async def test_second_identical_query_hits_the_cache():
    inner = CountingProvider()
    cached = CachedSearchProvider(inner, ttl_seconds=60.0)

    first = await cached.search("rust ownership")
    second = await cached.search("rust ownership")

    assert len(inner.calls) == 1
    assert second == first
    assert [r.url for r in second] == ["https://a.dev/x"]
    assert cached.hits == 1 and cached.misses == 1


@pytest.mark.asyncio
async def test_different_engines_bust_the_cache_key():
    inner = CountingProvider()
    cached = CachedSearchProvider(inner, ttl_seconds=60.0)

    await cached.search("rust ownership", engines=["duckduckgo"])
    await cached.search("rust ownership", engines=["brave"])
    await cached.search("rust ownership", engines=["duckduckgo"])

    assert len(inner.calls) == 2


@pytest.mark.asyncio
async def test_empty_results_are_never_cached():
    inner = CountingProvider(results=[])
    cached = CachedSearchProvider(inner, ttl_seconds=60.0)

    await cached.search("rust")
    await cached.search("rust")

    assert len(inner.calls) == 2


@pytest.mark.asyncio
async def test_expired_entries_are_refetched():
    inner = CountingProvider()
    cached = CachedSearchProvider(inner, ttl_seconds=0.0)

    await cached.search("rust")
    await cached.search("rust")

    assert len(inner.calls) == 2


@pytest.mark.asyncio
async def test_bypass_flag_forces_fresh_fetch():
    inner = CountingProvider()
    cached = CachedSearchProvider(inner, ttl_seconds=60.0)

    await cached.search("rust")
    with cache_bypass():
        await cached.search("rust")
    await cached.search("rust")

    assert len(inner.calls) == 2
    assert CACHE_BYPASS.get() is False
