import json

from src.core.tools.toolbox import AskToolbox, build_tools
from src.modules.search_engine.providers.base import SearchResult
from src.tools.settings import settings
from tests.core_ask.fakes import (
    FakeRanker,
    FakeScraper,
    FakeSearchProvider,
    make_scraper,
    sample_markdown,
)


def make_toolbox(scraper=None) -> AskToolbox:
    return AskToolbox(FakeSearchProvider(), scraper or make_scraper(), FakeRanker())


# <----- Search-result hygiene (AR-2 / WU-3) ----->


async def test_web_search_sanity_filter_on_by_default():
    # The exact Phase 3 poisoning: peptide results for a Rust query.
    provider = FakeSearchProvider(
        results=[
            SearchResult("Rust book", "ownership in rust", "https://a.dev/rust"),
            SearchResult("Peptides", "amino acids", "https://b.dev/peptides"),
        ]
    )
    toolbox = AskToolbox(provider, make_scraper(), FakeRanker())

    await toolbox.web_search("rust ownership")

    assert [s["url"] for s in toolbox.ranked_sources] == ["https://a.dev/rust"]
    assert toolbox.search_stats["dropped_results"] == 1


async def test_web_search_sanity_filter_off_keeps_everything(monkeypatch):
    monkeypatch.setattr(settings, "RESULT_SANITY_FILTER", False)
    provider = FakeSearchProvider(
        results=[
            SearchResult("Rust book", "ownership in rust", "https://a.dev/rust"),
            SearchResult("Peptides", "amino acids", "https://b.dev/peptides"),
        ]
    )
    toolbox = AskToolbox(provider, make_scraper(), FakeRanker())

    await toolbox.web_search("rust ownership")

    assert {s["url"] for s in toolbox.ranked_sources} == {
        "https://a.dev/rust",
        "https://b.dev/peptides",
    }
    assert toolbox.search_stats["dropped_results"] == 0


async def test_web_search_dedup_merges_engine_duplicates():
    provider = FakeSearchProvider(
        results=[
            SearchResult(
                "Short",
                "rust book page",
                "https://doc.rust-lang.org/book?utm_source=google",
            ),
            SearchResult(
                "The Rust Programming Language book",
                "a longer snippet about the rust book",
                "https://www.doc.rust-lang.org/book",
            ),
            SearchResult("Unrelated", "other rust page", "https://example.com/other"),
        ]
    )
    toolbox = AskToolbox(provider, make_scraper(), FakeRanker())

    await toolbox.web_search("rust book")

    urls = [s["url"] for s in toolbox.ranked_sources]
    assert urls == ["https://doc.rust-lang.org/book", "https://example.com/other"]
    merged_entry = toolbox.ranked_sources[0]
    # Longest title survives the merge (fixes the exact-URL title lookup).
    assert merged_entry["title"] == "The Rust Programming Language book"
    assert merged_entry["fusion_score"] > 0
    assert toolbox.search_stats["merged_results"] == 1


async def test_web_search_tiebreak_by_fusion():
    # Equal cross-encoder scores (FakeRanker: all 1.0); the duplicated
    # result has higher fusion (engine agreement) and must come first.
    provider = FakeSearchProvider(
        results=[
            SearchResult("Single", "rust one", "https://example.com/single"),
            SearchResult("Dup A", "rust two", "https://example.com/dup"),
            SearchResult(
                "Dup B", "rust three", "https://example.com/dup?utm_campaign=x"
            ),
        ]
    )
    toolbox = AskToolbox(provider, make_scraper(), FakeRanker())

    await toolbox.web_search("rust")

    assert [s["url"] for s in toolbox.ranked_sources] == [
        "https://example.com/dup",
        "https://example.com/single",
    ]


async def test_web_search_tiebreak_off_preserves_ranker_order(monkeypatch):
    monkeypatch.setattr(settings, "FUSION_TIEBREAK", False)
    provider = FakeSearchProvider(
        results=[
            SearchResult("Single", "rust one", "https://example.com/single"),
            SearchResult("Dup A", "rust two", "https://example.com/dup"),
            SearchResult(
                "Dup B", "rust three", "https://example.com/dup?utm_campaign=x"
            ),
        ]
    )
    toolbox = AskToolbox(provider, make_scraper(), FakeRanker())

    await toolbox.web_search("rust")

    # Ranker order (first-seen) stands; the merge still collapses duplicates.
    assert [s["url"] for s in toolbox.ranked_sources] == [
        "https://example.com/single",
        "https://example.com/dup",
    ]


async def test_web_search_max_urls_clamped():
    results = [
        SearchResult(f"title {i}", f"rust snippet {i}", f"https://example.com/{i}")
        for i in range(30)
    ]

    default_toolbox = AskToolbox(
        FakeSearchProvider(results=results), make_scraper(), FakeRanker()
    )
    await default_toolbox.web_search("rust")
    assert len(default_toolbox.ranked_sources) == settings.TOP_N_URLS

    single_toolbox = AskToolbox(
        FakeSearchProvider(results=results), make_scraper(), FakeRanker(), max_urls=1
    )
    await single_toolbox.web_search("rust")
    assert len(single_toolbox.ranked_sources) == 1
    assert single_toolbox.search_stats["max_urls"] == 1

    over_toolbox = AskToolbox(
        FakeSearchProvider(results=results), make_scraper(), FakeRanker(), max_urls=999
    )
    await over_toolbox.web_search("rust")
    assert len(over_toolbox.ranked_sources) == settings.TOP_N_URLS_MAX


async def test_web_search_fusion_floor_never_empties(monkeypatch):
    monkeypatch.setattr(settings, "URL_SELECTION_MIN_FUSION", 1.0)
    results = [
        SearchResult(f"title {i}", f"rust snippet {i}", f"https://example.com/{i}")
        for i in range(6)
    ]
    toolbox = AskToolbox(
        FakeSearchProvider(results=results), make_scraper(), FakeRanker()
    )

    await toolbox.web_search("rust")

    # Unreachable floor still keeps the three best-scoring results.
    assert len(toolbox.ranked_sources) == 3
    assert toolbox.search_stats["dropped_results"] == 3


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
    assert isinstance(scraper, FakeScraper)

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
