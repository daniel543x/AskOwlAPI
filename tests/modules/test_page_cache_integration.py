"""Page-cache repository round-trip (integration: live Postgres)."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from src.modules.scrapy.page_cache import PageCacheRepository, page_url_hash
from src.tools.settings import settings

pytestmark = pytest.mark.integration


async def test_put_then_get_fresh_roundtrip(db_session):
    repo = PageCacheRepository(db_session)

    await repo.put("https://Example.com/docs?utm_source=x", "<md> body", 200)
    fresh = await repo.get_fresh(["https://example.com/docs"])

    assert fresh == {"https://example.com/docs": "<md> body"}


async def test_expired_rows_are_not_served(db_session):
    repo = PageCacheRepository(db_session)
    await repo.put("https://a.dev/old", "stale body", 200)

    # Age the row beyond PAGE_CACHE_TTL_S directly in SQL.
    await db_session.execute(
        text("UPDATE scraped_page SET fetched_at = :old WHERE url_hash = :h"),
        {
            "old": datetime.now(UTC)
            - timedelta(seconds=settings.PAGE_CACHE_TTL_S + 1),
            "h": page_url_hash("https://a.dev/old"),
        },
    )
    await db_session.commit()

    assert await repo.get_fresh(["https://a.dev/old"]) == {}
