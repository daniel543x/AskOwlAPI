"""Provider health probing. Network-only; no database access here."""

import logging
from datetime import UTC, datetime

import httpx

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(UTC)


async def probe_provider(
    provider_type: str,
    base_url: str | None,
    timeout: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    """Probe one provider endpoint; returns a ProviderHealth value.

    - Ollama is probed via `/api/tags` (also proves the API is usable).
    - Other HTTP providers: any HTTP response counts as reachable.
    - Providers without a base_url are in-process ('local').
    """
    if not base_url:
        return "local"

    url = f"{base_url.rstrip('/')}/api/tags" if provider_type == "ollama" else base_url

    try:
        async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
            response = await client.get(url)
        # Any HTTP status means the endpoint answered.
        return "up" if response.status_code < 500 else "down"
    except httpx.RequestError:
        logger.debug("Probe failed for %s endpoint %s", provider_type, base_url)
        return "down"
