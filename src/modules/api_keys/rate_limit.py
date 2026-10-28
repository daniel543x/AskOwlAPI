"""Per-key, fixed-window rate limiting for API-key traffic.

Counters live in an `ICache` (in-process memory by default; Redis when
REDIS_URL is set), so the window semantics are backend-independent and a
multi-node deployment can share one window. The window anchors at a key's
first request and resets 60 s later. JWT traffic is not rate limited here.
"""

from typing import Any

from fastapi import HTTPException, status

from ...tools.cache import ICache

_WINDOW_SECONDS = 60.0


class RateLimiter:
    def __init__(self, limit_per_minute: int, cache: ICache):
        self.limit = limit_per_minute
        self._cache = cache

    async def check(self, key: Any) -> None:
        """Raise 429 when `key` exceeded the window limit; else count it."""
        cache_key = f"rate:{key}"
        count = await self._cache.incr(cache_key, _WINDOW_SECONDS)
        if count > self.limit:
            retry_after = int(await self._cache.ttl(cache_key)) + 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"API key rate limit exceeded ({self.limit} requests/minute)."
                ),
                headers={"Retry-After": str(retry_after)},
            )
