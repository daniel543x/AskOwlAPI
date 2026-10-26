"""Offline tests for the outbound egress proxy (PROXY_URL)."""

import httpx
import pytest

from src.modules.scrapy.providers.trafilatura import TrafilturaScraper
from src.modules.search_engine.providers.searxng import SearXNGProvider
from src.tools import http as http_tools
from src.tools.http import make_client, outbound_proxy
from src.tools.settings import settings


class ClientSpy(httpx.AsyncClient):
    """Real AsyncClient recording constructor kwargs."""

    captured: dict = {}

    def __init__(self, **kwargs):
        type(self).captured = dict(kwargs)
        super().__init__(**kwargs)


@pytest.fixture
def spy(monkeypatch):
    monkeypatch.setattr(http_tools.httpx, "AsyncClient", ClientSpy)
    ClientSpy.captured = {}
    return ClientSpy


def test_no_proxy_by_default():
    assert settings.PROXY_URL == ""
    assert outbound_proxy() is None


async def test_make_client_without_proxy_passes_no_proxy(spy):
    client = make_client(timeout=5.0)
    await client.aclose()

    assert "proxy" not in spy.captured
    assert spy.captured["timeout"] == 5.0


async def test_make_client_applies_configured_proxy(monkeypatch, spy):
    monkeypatch.setattr(settings, "PROXY_URL", "http://proxy.example:8080")

    assert outbound_proxy() == "http://proxy.example:8080"
    client = make_client(timeout=5.0)
    await client.aclose()

    assert spy.captured["proxy"] == "http://proxy.example:8080"


async def test_explicit_proxy_kwarg_is_not_overridden(monkeypatch, spy):
    monkeypatch.setattr(settings, "PROXY_URL", "http://default:1")

    client = make_client(proxy="http://explicit:2")
    await client.aclose()

    assert spy.captured["proxy"] == "http://explicit:2"


@pytest.mark.asyncio
async def test_searxng_search_goes_through_the_factory(monkeypatch, spy):
    monkeypatch.setattr(settings, "PROXY_URL", "http://proxy.example:8080")

    provider = SearXNGProvider(base_url="http://localhost:9")
    results = await provider.search("test")

    assert results == []  # connection refused degrades to no results
    assert spy.captured["proxy"] == "http://proxy.example:8080"
    assert spy.captured["timeout"] == provider.timeout


@pytest.mark.asyncio
async def test_scraper_fetch_goes_through_the_factory(monkeypatch, spy):
    monkeypatch.setattr(settings, "PROXY_URL", "http://proxy.example:8080")

    scraper = TrafilturaScraper()
    results = await scraper.scrape([{"url": "http://localhost:9/x"}])

    assert results == [{"content": "", "url": "http://localhost:9/x"}]
    assert spy.captured["proxy"] == "http://proxy.example:8080"
    assert spy.captured["follow_redirects"] is True
