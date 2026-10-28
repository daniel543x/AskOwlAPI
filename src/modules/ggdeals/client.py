"""gg.deals game-offers client (PR 14 / decision 18; optional, default off).

Structured price data for the `game_offers` agent tool: the LLM receives
compact JSON and the client renders an `offers` banner from the
structured payload — offers never enter answer text. Calls pass the
`ggdeals` resource gate and results cache through ICache.

The endpoint shape is verified at enablement time (the site is
Cloudflare-gated and the key is a paid tier), so the path is config:
with `GGDEALS_PRICES_PATH` unset the client degrades to a clear error
instead of guessing URLs. Response parsing is tolerant — the first
enablement tunes it against the real payload.
"""

import logging
from typing import Any

import httpx

from ...tools.cache import ICache, build_cache
from ...tools.settings import settings
from ..jobs.gates import GateKey, gated_turn

logger = logging.getLogger(__name__)

_GATE = "ggdeals"


class GgDealsClient:
    def __init__(self, cache: ICache | None = None):
        self._cache = cache or build_cache()

    def configured(self) -> bool:
        """True when the flag, key, and verified endpoint path are set."""
        return bool(
            settings.GGDEALS_ENABLED
            and settings.GGDEALS_API_KEY
            and settings.GGDEALS_PRICES_PATH
        )

    @staticmethod
    def _cache_key(game_name: str) -> str:
        normalized = " ".join(game_name.lower().split())
        return f"ggdeals:offers:{normalized}"

    async def game_offers(self, game_name: str) -> list[dict[str, Any]]:
        """Structured offers for one game; empty list on any failure.

        A cache hit skips the API entirely (TTL via
        GGDEALS_OFFERS_CACHE_TTL_S); every miss passes the `ggdeals`
        resource gate. Failures degrade to an empty result — a price
        lookup must never break an agent turn.
        """
        if not self.configured():
            return []
        key = self._cache_key(game_name)
        cached = await self._cache.get(key)
        if cached is not None:
            return list(cached)

        gate = GateKey(_GATE, settings.GGDEALS_MAX_CONCURRENCY)
        try:
            async with gated_turn([gate], wait_timeout=30.0):
                offers = await self._fetch(game_name)
        except Exception:
            logger.exception("gg.deals lookup failed for %r.", game_name)
            return []
        if offers:
            await self._cache.set(
                key, offers, settings.GGDEALS_OFFERS_CACHE_TTL_S
            )
        return offers

    async def _fetch(self, game_name: str) -> list[dict[str, Any]]:
        url = (
            f"{settings.GGDEALS_API_BASE_URL.rstrip('/')}"
            f"{settings.GGDEALS_PRICES_PATH}"
        )
        async with httpx.AsyncClient(timeout=settings.GGDEALS_TIMEOUT_S) as client:
            response = await client.get(
                url,
                params={"key": settings.GGDEALS_API_KEY, "title": game_name},
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            return parse_offers(response.json())


def _first(d: dict, *names: str) -> Any:
    for name in names:
        if name in d and d[name] is not None:
            return d[name]
    return None


def parse_offers(payload: Any) -> list[dict[str, Any]]:
    """Normalize an (unverified) API payload into the offer shape.

    Tolerant across the shapes the API documentation may settle on:
    a top-level list, `data`/`results` wrapped lists, or single-item
    dicts. Offers missing both a title and a price are dropped.
    """
    if isinstance(payload, dict):
        payload = _first(payload, "data", "results", "offers", "games")
        while isinstance(payload, dict):
            payload = _first(
                payload, "results", "data", "offers", "games", "items"
            )
    if not isinstance(payload, list):
        return []

    offers: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        title = _first(item, "title", "name", "game")
        price = _first(item, "price", "current", "best_price")
        historical = _first(item, "historical_low", "lowest", "historical")
        stores = _first(item, "stores", "shop", "store")
        if title is None and price is None:
            continue
        offers.append(
            {
                "title": str(title) if title is not None else "",
                "current_best": price,
                "historical_low": historical,
                "stores": stores,
            }
        )
    return offers
