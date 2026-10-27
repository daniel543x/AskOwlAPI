"""In-memory TTL cache wrapping any ISearchProvider (WU-10 / AR-1).

Per-process memoization: an identical (query, engines) pair within the TTL
skips the upstream SearXNG round-trip. Empty responses are never cached —
a transient upstream failure must not be frozen in. One shared instance per
provider configuration lives in `SearchProviderFactory` (the single choke
point), so ask, research, agent, and eval all benefit with zero pipeline
edits.
"""

import hashlib
import json
import logging
import time
from collections import OrderedDict

from ...tools.cache_flags import CACHE_BYPASS
from .providers.base import ISearchProvider, SearchResult

logger = logging.getLogger(__name__)


class CachedSearchProvider(ISearchProvider):
    """LRU + TTL memoizing decorator around an inner search provider."""

    def __init__(
        self,
        inner: ISearchProvider,
        ttl_seconds: float,
        max_entries: int = 512,
    ):
        self.inner = inner
        self.ttl = ttl_seconds
        self.max = max_entries
        self._store: OrderedDict[str, tuple[float, list[SearchResult]]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _key(self, query: str, engines: list[str] | None) -> str:
        raw = json.dumps([query, sorted(engines or [])], sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    async def search(
        self, query: str, engines: list[str] | None = None
    ) -> list[SearchResult]:
        if CACHE_BYPASS.get():
            return list(await self.inner.search(query, engines=engines))

        key = self._key(query, engines)
        now = time.monotonic()
        hit = self._store.get(key)
        if hit is not None and now - hit[0] < self.ttl:
            self._store.move_to_end(key)
            self.hits += 1
            logger.debug("Search cache hit for %r.", query)
            return list(hit[1])

        results = await self.inner.search(query, engines=engines)
        self.misses += 1
        if results:
            self._store[key] = (now, list(results))
            self._store.move_to_end(key)
            while len(self._store) > self.max:
                self._store.popitem(last=False)
        return list(results)
