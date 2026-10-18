"""Per-key, in-process, fixed-window rate limiting for API-key traffic.

Single-node by design (AskOwl deploys as one container); swap for Redis
when a multi-node topology appears. JWT traffic is not rate limited here.
"""

import time
from typing import Any

from fastapi import HTTPException, status


class RateLimiter:
    def __init__(self, limit_per_minute: int):
        self.limit = limit_per_minute
        self._windows: dict[Any, tuple[float, int]] = {}

    def check(self, key: Any) -> None:
        """Raise 429 when `key` exceeded the window limit; else count it."""
        now = time.monotonic()
        window_start, count = self._windows.get(key, (now, 0))
        if now - window_start >= 60.0:
            window_start, count = now, 0
        if count >= self.limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"API key rate limit exceeded ({self.limit} requests/minute)."
                ),
                headers={"Retry-After": str(int(60 - (now - window_start)) + 1)},
            )
        self._windows[key] = (window_start, count + 1)
