"""Outbound HTTP client factory for search/scrape egress.

All search-engine and page-fetching traffic goes through `make_client` so
the optional `PROXY_URL` applies uniformly — hosting providers where
direct traffic from app hosts is blocked can route egress through a proxy
with one setting. LLM/embedding provider clients intentionally stay
direct: they frequently target localhost (Ollama, LAN boxes).
"""

from typing import Any

import httpx

from .settings import settings


def outbound_proxy() -> str | None:
    """Configured egress proxy, or None for direct connections."""
    return settings.PROXY_URL or None


def make_client(**kwargs: Any) -> httpx.AsyncClient:
    """httpx.AsyncClient with the egress proxy applied when configured."""
    proxy = outbound_proxy()
    if proxy:
        kwargs.setdefault("proxy", proxy)
    return httpx.AsyncClient(**kwargs)
