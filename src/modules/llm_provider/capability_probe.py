"""Provider capability probing with ICache-backed results.

Precedence per catalog entry (decision 20): **admin override > probe >
seed**. The admin override is a non-empty `effort_levels` (or
`capabilities`) list on the catalog entry; the probe result lives in the
ICache under a per-entry key with a TTL; the seed is the built-in table
below. A probe may only prune confirmed-unsupported levels, never add,
and a failed probe keeps the seed (degrade, never crash). Probes are
scoped to a single catalog entry — a result never changes what other
providers advertise.

Ollama is probed live via `/api/show` (model capabilities) and `/api/ps`
(loaded models); its think levels are request-level, so they stay as
seeded and are pruned by the request-time fallback in the adapters.
z.ai levels are probed with one synthetic call per seeded level
(OpenAI/OpenRouter stay seed/metadata-driven — no paid-API probing).
"""

import logging
from typing import Any

import httpx

from ...tools.cache import ICache, build_cache
from ...tools.settings import settings
from .selection import ResolvedModel

logger = logging.getLogger(__name__)

# Built-in effort-level seeds per provider type (decision 20): GLM-5.2+
# ship [low, medium, high, max]; every other provider starts empty and is
# admin-seeded per model.
SEED_EFFORT_LEVELS: dict[str, tuple[str, ...]] = {
    "zai": ("low", "medium", "high", "max"),
}

_PROBE_TIMEOUT_S = 5.0


def probe_cache_key(entry_id: str) -> str:
    """ICache key for one catalog entry's probe result."""
    return f"capabilities:{entry_id}"


def _seeded_levels(provider_type: str) -> list[str]:
    return list(SEED_EFFORT_LEVELS.get(provider_type.lower(), ()))


async def _probe_ollama(
    base_url: str, model_name: str
) -> dict[str, Any] | None:
    """Capabilities of one Ollama model via /api/show; None on any failure."""
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_S) as client:
            response = await client.post(
                f"{base_url.rstrip('/')}/api/show", json={"model": model_name}
            )
            response.raise_for_status()
            data = response.json()
    except (httpx.HTTPError, ValueError):
        logger.debug(
            "Ollama capability probe failed for %s; keeping the seed.",
            model_name,
        )
        return None
    capabilities = data.get("capabilities")
    if not isinstance(capabilities, list):
        return None
    return {
        "capabilities": [str(item) for item in capabilities],
        "effort_levels": [],
    }


async def _probe_zai_levels(
    base_url: str, api_key: str | None, model_name: str, seeds: list[str]
) -> list[str] | None:
    """Seed levels that survive one tiny synthetic call each; None on
    infrastructure failure (keep the seed). Rejected levels are pruned."""
    kept: list[str] = []
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_S) as client:
            for level in seeds:
                response = await client.post(
                    url,
                    headers=headers,
                    json={
                        "model": model_name,
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 1,
                        "thinking": {"type": "enabled"},
                        "reasoning_effort": level,
                    },
                )
                # 4xx for a level = confirmed-unsupported (or aliased, which
                # the probe cannot distinguish — the conservative variant
                # wins by pruning). 5xx / network errors keep the level.
                if response.status_code < 400:
                    kept.append(level)
    except httpx.HTTPError:
        logger.debug(
            "z.ai effort-level probe failed for %s; keeping the seed.",
            model_name,
        )
        return None
    return kept


async def effective_reasoning_state(
    resolved: ResolvedModel,
    cache: ICache | None = None,
) -> tuple[list[str], list[str]]:
    """Capabilities + effort levels for gating one resolved entry.

    Precedence: catalog values (admin) > probe cache / fresh probe > seed.
    Never raises: a probe error falls back to the catalog or seed values.
    """
    capabilities = list(resolved.capabilities)
    levels = list(resolved.effort_levels)
    if levels:
        # Admin override wins untouched — the admin may knowingly advertise
        # levels the seed table or probe would prune.
        return capabilities, levels

    provider = resolved.provider_type.lower()
    if capabilities and not _seeded_levels(provider):
        # Admin-curated capabilities, no seed levels: nothing to learn.
        return capabilities, levels

    cache = cache or build_cache()
    probed: dict[str, Any] | None = None
    if settings.CAPABILITY_PROBE_TTL_S > 0:
        key = probe_cache_key(str(resolved.entry_id))
        probed = await cache.get(key)
        if probed is None:
            if provider == "ollama" and resolved.base_url:
                probed = await _probe_ollama(resolved.base_url, resolved.model_name)
            elif provider == "zai":
                kept = await _probe_zai_levels(
                    resolved.base_url or "",
                    resolved.api_key,
                    resolved.model_name,
                    _seeded_levels(provider),
                )
                if kept is not None:
                    probed = {"capabilities": [], "effort_levels": kept}
            if probed is not None:
                await cache.set(key, probed, settings.CAPABILITY_PROBE_TTL_S)

    if probed:
        if not capabilities and probed.get("capabilities"):
            # Probe facts fill the gap; a probe never removes admin values.
            capabilities = list(probed["capabilities"])
        if not levels and probed.get("effort_levels"):
            levels = list(probed["effort_levels"])

    if not levels:
        levels = _seeded_levels(provider)
    return capabilities, levels
