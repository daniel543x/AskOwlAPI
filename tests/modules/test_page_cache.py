"""Offline tests for the per-URL page cache wrapper (WU-10 / AR-1)."""

import pytest

from src.modules.scrapy.page_cache import PageCachingScraper, page_url_hash
from src.tools.cache_flags import cache_bypass
from src.tools.settings import settings
from tests.core_ask.fakes import FakeScraper


class FakePageRepo:
    """In-memory stand-in for PageCacheRepository."""

    def __init__(self, fresh: dict[str, str] | None = None):
        self.fresh = fresh or {}
        self.written: list[tuple[str, str]] = []

    async def get_fresh(self, urls):
        return {u: c for u, c in self.fresh.items() if u in set(urls)}

    async def put(self, url, content, http_status):
        self.written.append((url, content))


@pytest.mark.asyncio
async def test_cache_hit_skips_the_fetch():
    inner = FakeScraper(pages={"https://a.dev/x": "fresh"})
    repo = FakePageRepo(fresh={"https://a.dev/x": "cached content"})
    scraper = PageCachingScraper(inner, repo)

    pages = await scraper.scrape([{"url": "https://a.dev/x"}])

    assert pages == [{"content": "cached content", "url": "https://a.dev/x"}]
    assert inner.requested == []
    assert scraper.hits == 1 and scraper.misses == 0


@pytest.mark.asyncio
async def test_miss_fetches_then_writes_the_cache():
    inner = FakeScraper(pages={"https://a.dev/x": "fresh body"})
    repo = FakePageRepo()
    scraper = PageCachingScraper(inner, repo)

    pages = await scraper.scrape([{"url": "https://a.dev/x"}])

    assert pages[0]["content"] == "fresh body"
    assert inner.requested == ["https://a.dev/x"]
    assert repo.written == [("https://a.dev/x", "fresh body")]


@pytest.mark.asyncio
async def test_tracking_param_variant_hits_the_same_entry():
    inner = FakeScraper(pages={})
    # The repo keys by the requested URL (hash mapping lives in the repo);
    # the variant URL normalizes to the same hash, so it is served fresh.
    repo = FakePageRepo(fresh={"https://www.a.dev/x?utm_source=b": "cached"})
    scraper = PageCachingScraper(inner, repo)

    pages = await scraper.scrape([{"url": "https://www.a.dev/x?utm_source=b"}])

    assert pages[0]["content"] == "cached"
    assert inner.requested == []
    assert page_url_hash("https://a.dev/x") == page_url_hash(
        "https://www.a.dev/x?utm_source=b"
    )


@pytest.mark.asyncio
async def test_bypass_flag_forces_fresh_fetch():
    inner = FakeScraper(pages={"https://a.dev/x": "fresh body"})
    repo = FakePageRepo(fresh={"https://a.dev/x": "cached"})
    scraper = PageCachingScraper(inner, repo)

    with cache_bypass():
        pages = await scraper.scrape([{"url": "https://a.dev/x"}])

    assert pages[0]["content"] == "fresh body"
    assert inner.requested == ["https://a.dev/x"]
    assert repo.written == []


@pytest.mark.asyncio
async def test_disabled_cache_fetches_fresh(monkeypatch):
    monkeypatch.setattr(settings, "PAGE_CACHE", False)
    inner = FakeScraper(pages={"https://a.dev/x": "fresh body"})
    repo = FakePageRepo(fresh={"https://a.dev/x": "cached"})
    scraper = PageCachingScraper(inner, repo)

    pages = await scraper.scrape([{"url": "https://a.dev/x"}])

    assert pages[0]["content"] == "fresh body"
    assert repo.written == []


@pytest.mark.asyncio
async def test_order_preserved_and_empty_fetch_degrades():
    inner = FakeScraper(pages={"https://b.dev/y": ""})  # empty extraction
    repo = FakePageRepo(fresh={"https://a.dev/x": "cached a"})
    scraper = PageCachingScraper(inner, repo)

    pages = await scraper.scrape(
        [{"url": "https://b.dev/y"}, {"url": "https://a.dev/x"}]
    )

    assert [p["url"] for p in pages] == ["https://b.dev/y", "https://a.dev/x"]
    assert pages[0]["content"] == ""
    assert pages[1]["content"] == "cached a"


def test_page_url_hash_uses_normalized_url():
    assert page_url_hash("https://Example.com/a/?utm_source=x") == page_url_hash(
        "https://example.com/a"
    )
    assert page_url_hash("https://example.com/a") != page_url_hash(
        "https://example.com/b"
    )
