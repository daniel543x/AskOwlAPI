import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

import httpx
import trafilatura

from ....tools.http import make_client
from ....tools.settings import settings
from .base import IScraper

logger = logging.getLogger(__name__)

# Realistic browser UA to avoid the cheapest bot filters.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class TrafilturaScraper(IScraper):
    """Fetches pages concurrently with httpx and extracts content with trafilatura.

    Network I/O runs through a semaphore-bounded AsyncClient; CPU-bound
    extraction runs in a thread pool so the event loop is never blocked.
    """

    def __init__(
        self, api_key: str | None = None, config: dict | None = None
    ) -> None:
        super().__init__(api_key=api_key, config=config)
        config = self.config or {}
        self.concurrency = int(config.get("concurrency", settings.SCRAPER_CONCURRENCY))
        self.timeout = float(config.get("timeout", settings.SCRAPER_TIMEOUT))
        self.user_agent = str(config.get("user_agent", DEFAULT_USER_AGENT))

    @property
    def name(self) -> str:
        return "trafilatura"

    def _build_client(self) -> httpx.AsyncClient:
        # Shared factory: applies the optional egress proxy (PROXY_URL).
        return make_client(
            timeout=self.timeout,
            headers={"User-Agent": self.user_agent},
            follow_redirects=True,
        )

    async def scrape(
        self, data_from_search: list[dict[str, Any]]
    ) -> list[dict[str, Any]] | None:
        if not data_from_search:
            return []

        semaphore = asyncio.Semaphore(self.concurrency)
        async with self._build_client() as client:
            results = await asyncio.gather(
                *(
                    self._scrape_one(client, semaphore, item)
                    for item in data_from_search
                )
            )
        return list(results)

    async def _scrape_one(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        item: dict[str, Any],
    ) -> dict[str, Any]:
        url = item.get("url", "")
        try:
            async with semaphore:
                response = await client.get(url)
            if response.status_code != 200:
                logger.warning(
                    "Failed to fetch '%s': HTTP %d", url, response.status_code
                )
                return {"content": "", "url": url}

            parts = await asyncio.to_thread(self._extract_parts, response.text)
            return {
                "content": parts["content"],
                "url": url,
                "title": parts["title"],
                "published_at": parts["published_at"],
                "fetched_at": datetime.now(UTC).isoformat(),
            }
        except Exception:
            logger.warning("Failed to scrape '%s'", url, exc_info=True)
            return {"content": "", "url": url}

    @staticmethod
    def _extract_parts(html: str) -> dict[str, Any]:
        """Markdown content plus page metadata (AR-7 task 2).

        Two passes: `extract` for the markdown body (bare_extraction does
        not apply markdown output to its text field), `bare_extraction`
        for the metadata (title, publication date).
        """
        markdown = (
            trafilatura.extract(
                html,
                include_links=True,
                include_images=False,
                output_format="markdown",
            )
            or ""
        )
        if not markdown:
            return {"content": "", "title": "", "published_at": None}

        extracted = trafilatura.bare_extraction(html, with_metadata=True)
        if not extracted:
            return {"content": markdown, "title": "", "published_at": None}

        def field(name: str) -> Any:
            # Newer trafilatura returns a Document (attributes), older
            # versions a dict — support both access styles.
            value: Any = getattr(extracted, name, None)
            if value is None and hasattr(extracted, "get"):
                value = extracted.get(name)  # pyright: ignore[reportAttributeAccessIssue]
            return value

        raw_date = str(field("date") or "") or None
        published_at = None
        if raw_date:
            try:
                published_at = datetime.fromisoformat(raw_date[:10]).date().isoformat()
            except ValueError:
                published_at = None
        return {
            "content": markdown,
            "title": str(field("title") or ""),
            "published_at": published_at,
        }

    @staticmethod
    def _extract(html: str) -> str:
        extracted = trafilatura.extract(
            html,
            include_links=True,
            include_images=False,
            output_format="markdown",
        )
        return extracted or ""
