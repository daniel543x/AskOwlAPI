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
    assert result is not None

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
    assert result is not None

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


# <----- Page metadata extraction (AR-7 task 2) ----->

METADATA_HTML = """<html><head><title>Metadata Test Page</title>
<meta name="date" content="2026-01-15">
</head><body>
<h1>Metadata Header</h1>
<p>Enough meaningful text for the extractor to keep this page.</p>
</body></html>"""


async def test_scrape_carries_title_and_publication_date(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=METADATA_HTML)

    _install_mock_transport(monkeypatch, handler)
    scraper = TrafilturaScraper()

    result = await scraper.scrape([{"url": "http://meta.test/1"}])

    assert result is not None
    page = result[0]
    # trafilatura reports the extracted content heading as the title.
    assert page["title"] == "Metadata Header"
    assert page["published_at"] == "2026-01-15"
    assert page["fetched_at"]


async def test_chunk_metadata_carries_title_site_and_dates():
    from src.core.tools.chunking import chunk_markdown_data

    chunks = chunk_markdown_data(
        [
            {
                "url": "https://docs.example/guide/intro",
                "content": "# Guide\n\nSome section text long enough to chunk.",
                "title": "The Guide",
                "published_at": "2026-01-15",
                "fetched_at": "2026-10-26T00:00:00+00:00",
            }
        ]
    )

    assert chunks
    for chunk in chunks:
        metadata = chunk["metadata"]
        assert metadata["title"] == "The Guide"
        assert metadata["site"] == "docs.example"
        assert metadata["published_at"] == "2026-01-15"
        assert metadata["fetched_at"] == "2026-10-26T00:00:00+00:00"


def test_registrable_domain():
    from src.tools.urls import registrable_domain

    assert registrable_domain("https://docs.example/guide") == "docs.example"
    assert registrable_domain("https://a.b.example/x") == "b.example"
    assert registrable_domain("https://www.example.com") == "example.com"
    assert registrable_domain("http://localhost:3000/x") == "localhost"
