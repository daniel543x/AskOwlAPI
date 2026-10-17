import httpx
import pytest

from src.modules.scrapy.providers.trafilatura import TrafilturaScraper

SAMPLE_HTML = """<html><head><title>Test Page</title></head><body>
<h1>Unique Header One</h1>
<p>This is the first paragraph with plenty of meaningful text for the extractor.</p>
<h2>Unique Header Two</h2>
<p>This is the second paragraph discussing the subsection in enough detail here.</p>
</body></html>"""


def _install_mock_transport(monkeypatch, handler) -> None:
    def fake_build_client(self: TrafilturaScraper) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            timeout=self.timeout,
            headers={"User-Agent": self.user_agent},
            follow_redirects=True,
        )

    monkeypatch.setattr(TrafilturaScraper, "_build_client", fake_build_client)


async def test_scrape_extracts_content_from_all_pages(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert "User-Agent" in request.headers
        return httpx.Response(200, text=SAMPLE_HTML)

    _install_mock_transport(monkeypatch, handler)
    scraper = TrafilturaScraper(config={"concurrency": 2})

    result = await scraper.scrape(
        [{"url": "http://a.test/1"}, {"url": "http://a.test/2"}, {"url": "http://a.test/3"}]
    )

    assert [item["url"] for item in result] == [
        "http://a.test/1",
        "http://a.test/2",
        "http://a.test/3",
    ]
    for item in result:
        assert "plenty of meaningful text" in item["content"]
        assert item["url"].startswith("http://a.test/")  # url is a string, not a dict


async def test_scrape_returns_empty_content_on_http_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if "fail" in str(request.url):
            return httpx.Response(500)
        return httpx.Response(200, text=SAMPLE_HTML)

    _install_mock_transport(monkeypatch, handler)
    scraper = TrafilturaScraper()

    result = await scraper.scrape([{"url": "http://ok.test/1"}, {"url": "http://fail.test/x"}])

    assert result[0]["content"] != ""
    assert result[1] == {"content": "", "url": "http://fail.test/x"}


async def test_scrape_handles_connection_errors(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    _install_mock_transport(monkeypatch, handler)
    scraper = TrafilturaScraper()

    result = await scraper.scrape([{"url": "http://down.test/1"}])

    assert result == [{"content": "", "url": "http://down.test/1"}]


async def test_scrape_empty_input_returns_empty_list():
    scraper = TrafilturaScraper()
    assert await scraper.scrape([]) == []


def test_scraper_reads_config_overrides():
    scraper = TrafilturaScraper(config={"concurrency": 9, "timeout": 1.5})
    assert scraper.concurrency == 9
    assert scraper.timeout == 1.5
    assert scraper.name == "trafilatura"


def test_scraper_falls_back_to_settings_defaults():
    from src.tools.settings import settings

    scraper = TrafilturaScraper()
    assert scraper.concurrency == settings.SCRAPER_CONCURRENCY
    assert scraper.timeout == settings.SCRAPER_TIMEOUT


@pytest.mark.parametrize("missing", [None, {}])
def test_scraper_treats_missing_config_as_empty(missing):
    scraper = TrafilturaScraper(config=missing)
    assert scraper.concurrency > 0
