"""Per-URL scraped-content cache (WU-10 / AR-1).

Postgres-backed so it survives restarts. A cache hit skips both the HTTP
fetch and the `trafilatura` extraction. Keys are the sha256 of the
normalized URL (PR 1), so tracking-param variants share one entry. Only
successful extractions (non-empty content) are stored — a transient fetch
failure is retried on the next request instead of being frozen in.
"""

import hashlib
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from ...tools.cache_flags import CACHE_BYPASS
from ...tools.settings import settings
from ...tools.urls import normalize_url
from .providers.base import IScraper

logger = logging.getLogger(__name__)


def page_url_hash(url: str) -> str:
    """Cache key: sha256 of the normalized URL."""
    return hashlib.sha256(normalize_url(url).encode()).hexdigest()


class PageCacheRepoLike(Protocol):
    """Repo surface the scraper wrapper needs; PageCacheRepository satisfies it."""

    async def get_fresh(self, urls: list[str]) -> dict[str, str]: ...

    async def put(self, url: str, content: str, http_status: int | None) -> None: ...


class PageCacheRepository:
    """DB access for the scraped-page cache."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_fresh(self, urls: list[str]) -> dict[str, str]:
        """Content for the URLs cached within the TTL, keyed by requested URL.

        The cache stores rows under the normalized-URL hash, but callers
        speak in original URLs — the mapping back happens here so variants
        (tracking params, `www.`) hit the same entry.
        """
        if not urls:
            return {}
        hashes_by_url = {url: page_url_hash(url) for url in urls}
        cutoff = datetime.now(UTC) - timedelta(seconds=settings.PAGE_CACHE_TTL_S)
        statement = text(
            "SELECT url_hash, content FROM scraped_page "
            "WHERE url_hash = ANY(:hashes) AND fetched_at > :cutoff"
        )
        try:
            result = await self.session.execute(
                statement, {"hashes": list(hashes_by_url.values()), "cutoff": cutoff}
            )
        except SQLAlchemyError:
            # Cache read failures degrade to a full fetch, never break a turn.
            await self.session.rollback()
            logger.warning("Page-cache read failed; fetching fresh.", exc_info=True)
            return {}
        content_by_hash = {row.url_hash: row.content for row in result}
        return {
            url: content_by_hash[h]
            for url, h in hashes_by_url.items()
            if h in content_by_hash
        }

    async def put(self, url: str, content: str, http_status: int | None) -> None:
        """Upsert one scraped page; failures degrade silently."""
        if not content:
            return
        statement = text(
            "INSERT INTO scraped_page (url_hash, url, content, http_status, fetched_at)"
            " VALUES (:url_hash, :url, :content, :http_status, :fetched_at)"
            " ON CONFLICT (url_hash) DO UPDATE SET url = :url, content = :content,"
            " http_status = :http_status, fetched_at = :fetched_at"
        )
        try:
            await self.session.execute(
                statement,
                {
                    "url_hash": page_url_hash(url),
                    "url": normalize_url(url),
                    "content": content,
                    "http_status": http_status,
                    "fetched_at": datetime.now(UTC),
                },
            )
            await self.session.commit()
        except SQLAlchemyError:
            await self.session.rollback()
            logger.warning("Page-cache write failed; ignoring.", exc_info=True)


class PageCachingScraper(IScraper):
    """Scraper decorator serving cached page content within the TTL."""

    def __init__(self, inner: IScraper, repo: PageCacheRepoLike):
        self.inner = inner
        self.repo = repo
        self.hits = 0
        self.misses = 0

    @property
    def name(self) -> str:
        return f"page-cached({self.inner.name})"

    async def scrape(
        self, data_from_search: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not data_from_search:
            return []
        if CACHE_BYPASS.get() or not settings.PAGE_CACHE:
            return (await self.inner.scrape(data_from_search)) or []

        targets = [item.get("url", "") for item in data_from_search]
        cached = await self.repo.get_fresh(targets)
        misses = [item for item in data_from_search if item.get("url") not in cached]
        self.hits += len(targets) - len(misses)
        self.misses += len(misses)

        fetched: list[dict[str, Any]] = []
        if misses:
            fetched = await self.inner.scrape(misses) or []
            for page in fetched:
                if page.get("url") and page.get("content"):
                    await self.repo.put(
                        page["url"], page["content"], page.get("http_status")
                    )

        # Preserve the request order; misses that came back empty degrade to
        # the same shape the pipeline already handles.
        by_url = {page.get("url"): page for page in fetched}
        return [
            {"content": cached[item["url"]], "url": item["url"]}
            if item.get("url") in cached
            else by_url.get(item.get("url"), {"content": "", "url": item.get("url")})
            for item in data_from_search
        ]
