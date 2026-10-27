"""Shared cache-bypass flag for the hot-path caches (WU-10).

The `cache: "bypass"` request parameter must reach the search-provider and
page-cache wrappers even though they are constructed before the request body
is parsed, so the flag travels through a context variable that both wrappers
read at call time. Adapters set it around the pipeline (never at import
time), and it is unset on every error path via the context manager.
"""

import contextvars
from collections.abc import Generator
from contextlib import contextmanager

CACHE_BYPASS: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "cache_bypass", default=False
)


@contextmanager
def cache_bypass() -> Generator[None]:
    """Force fresh search + scrape for everything inside the block."""
    token = CACHE_BYPASS.set(True)
    try:
        yield
    finally:
        CACHE_BYPASS.reset(token)
