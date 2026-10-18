import json

from src.core.ask.services.tools import AskToolbox, build_tools
from tests.core_ask.fakes import (
    FakeRanker,
    FakeScraper,
    FakeSearchProvider,
    make_scraper,
    sample_markdown,
)


def make_toolbox(scraper=None) -> AskToolbox:
    return AskToolbox(FakeSearchProvider(), scraper or make_scraper(), FakeRanker())


async def test_web_search_returns_ranked_json_and_stores_state():
    toolbox = make_toolbox()

    result = json.loads(await toolbox.web_search("rust language"))

    assert [item["url"] for item in result] == [
        "http://example.com/a",
        "http://example.com/b",
    ]
    assert result[0]["title"] == "Result A"
    assert result[0]["score"] == 1.0
    assert toolbox.ranked_sources == result
    assert len(toolbox.raw_results) == 2


async def test_web_search_empty_results():
    toolbox = AskToolbox(
        FakeSearchProvider(results=[]), FakeScraper(), FakeRanker()
    )

    result = json.loads(await toolbox.web_search("anything"))

    assert result == []
    assert toolbox.ranked_sources == []


async def test_fetch_pages_defaults_to_ranked_sources_and_chunks():
    toolbox = make_toolbox()
    await toolbox.web_search("rust")
    scraper = toolbox.scraper

    preview = await toolbox.fetch_pages()

    assert scraper.requested == [
        "http://example.com/a",
        "http://example.com/b",
    ]
    assert len(toolbox.chunks) == 4  # two pages x two header sections
    assert "first paragraph" in preview
    assert "http://example.com/a" in preview


async def test_fetch_pages_with_explicit_urls():
    scraper = FakeScraper(pages={"http://x/1": sample_markdown()})
    toolbox = AskToolbox(FakeSearchProvider(), scraper, FakeRanker())

    await toolbox.fetch_pages(["http://x/1"])

    assert scraper.requested == ["http://x/1"]
    assert len(toolbox.chunks) == 2


async def test_fetch_pages_without_urls_or_results():
    toolbox = AskToolbox(
        FakeSearchProvider(results=[]), FakeScraper(), FakeRanker()
    )

    preview = await toolbox.fetch_pages()

    assert preview == "No URLs to fetch."
    assert toolbox.chunks == []


async def test_get_context_ranks_chunks_and_joins():
    toolbox = make_toolbox()
    await toolbox.web_search("rust")
    await toolbox.fetch_pages()

    context = await toolbox.get_context("rust")

    assert "first paragraph" in context
    assert "second paragraph" in context
    assert "---" in context
    assert len(toolbox.selected_chunks) == len(toolbox.chunks)


async def test_get_context_respects_top_n():
    toolbox = make_toolbox()
    toolbox.chunks = [{"content": f"chunk {i}"} for i in range(50)]

    context = await toolbox.get_context("q")

    from src.tools.settings import settings

    assert len(toolbox.selected_chunks) == settings.TOP_N_CHUNKS
    assert context.count("---") == settings.TOP_N_CHUNKS - 1


async def test_ranker_receives_expected_calls():
    ranker = FakeRanker()
    toolbox = AskToolbox(FakeSearchProvider(), make_scraper(), ranker)
    from src.tools.settings import settings

    await toolbox.web_search("rust")
    await toolbox.fetch_pages()
    await toolbox.get_context("rust")

    assert ("web", "rust", 2, settings.TOP_N_URLS) in ranker.calls
    assert any(
        call[0] == "chunks" and call[3] == settings.TOP_N_CHUNKS
        for call in ranker.calls
    )


def test_build_tools_exposes_three_named_tools():
    tools = build_tools(make_toolbox())

    assert [tool.name for tool in tools] == ["web_search", "fetch_pages", "get_context"]
    assert all(tool.coroutine is not None for tool in tools)
