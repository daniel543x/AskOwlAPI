import logging

import httpx

from ....tools.settings import settings
from .base import ISearchProvider, SearchResult

logger = logging.getLogger(__name__)


class SearXNGProvider(ISearchProvider):
    """SearXNG metasearch client using its JSON API.

    All connection parameters are constructor arguments so they can be wired
    from database settings or environment variables, never hardcoded.
    """

    def __init__(
        self,
        base_url: str | None = None,
        engines: str | None = None,
        categories: str | None = None,
        timeout: float | None = None,
    ):
        self.base_url = base_url or settings.SEARXNG_BASE_URL
        self.engines = engines or settings.SEARXNG_ENGINES
        self.categories = categories or settings.SEARXNG_CATEGORIES
        self.timeout = timeout if timeout is not None else settings.SEARXNG_TIMEOUT

    async def search(self, query: str) -> list[SearchResult]:
        params = {
            "q": query,
            "format": "json",
            "engines": self.engines,
            "categories": self.categories,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/search", params=params)
                response.raise_for_status()
                results = response.json().get("results", [])

                return [
                    SearchResult(
                        title=item.get("title", ""),
                        snippet=item.get("content", ""),
                        url=item.get("url", ""),
                    )
                    for item in results
                ]
        except httpx.RequestError:
            logger.exception("SearXNG request failed (base_url=%s)", self.base_url)
            return []
