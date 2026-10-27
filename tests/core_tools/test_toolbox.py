import json
import uuid
from datetime import UTC, datetime

from src.core.tools.toolbox import AskToolbox, build_tools
from src.modules.chunk_store.model import chunk_hash
from src.modules.search_engine.providers.base import SearchResult
from src.tools.cache_flags import CACHE_BYPASS, cache_bypass
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


# <----- Query fan-out (WU-2) and hot-path caches (WU-10) ----->


class PerQuerySearchProvider(FakeSearchProvider):
    """Returns different results per query, recording every call."""

    def __init__(self, by_query: dict[str, list[SearchResult]]):
        super().__init__(results=[])
        self.by_query = by_query
        self.queries: list[str] = []

    async def search(self, query, engines=None):
        self.queries.append(query)
        return self.by_query.get(query, [])


async def test_web_search_many_fans_out_and_merges_through_dedup():
    provider = PerQuerySearchProvider(
        {
            "rust ownership": [
                SearchResult("Book", "rust book", "https://doc.rust-lang.org/book")
            ],
            "rust ownership explained": [
                SearchResult(
                    "Book (mirror)",
                    "the rust book page",
                    "https://www.doc.rust-lang.org/book?utm_source=x",
                ),
                SearchResult(
                    "Borrow checker", "rust borrow checker", "https://b.dev/borrow"
                ),
            ],
        }
    )
    toolbox = AskToolbox(provider, make_scraper(), FakeRanker())

    await toolbox.web_search_many(["rust ownership", "rust ownership explained"])

    assert provider.queries == ["rust ownership", "rust ownership explained"]
    # Both variants' URLs survive, the duplicate page merged into one entry.
    urls = sorted(s["url"] for s in toolbox.ranked_sources)
    assert urls == ["https://b.dev/borrow", "https://doc.rust-lang.org/book"]
    assert toolbox.search_stats["merged_results"] == 1


class BypassAwareProvider(FakeSearchProvider):
    """Records whether the cache-bypass flag was set during search."""

    def __init__(self):
        super().__init__()
        self.bypass_seen: list[bool] = []

    async def search(self, query, engines=None):
        self.bypass_seen.append(CACHE_BYPASS.get())
        return self.results


async def test_cache_bypass_flag_reaches_the_search_provider():
    provider = BypassAwareProvider()
    toolbox = AskToolbox(provider, make_scraper(), FakeRanker())

    await toolbox.web_search("rust")
    with cache_bypass():
        await toolbox.web_search("rust")

    assert provider.bypass_seen == [False, True]


# <----- Embed-skip (WU-10 / AR-1) ----->


class RecordingEmbedder:
    def __init__(self):
        self.embedded: list[str] = []

    @property
    def dims(self) -> int:
        return 2

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [[1.0, 0.0] for _ in texts]


class PreStoredChunkRepo:
    """Chunk store that reports pre-computed hashes as already stored."""

    def __init__(self, known_hashes: set[str]):
        self.known_hashes = known_hashes
        self.stored_chunks: list[dict] = []

    async def existing_hashes(self, hashes: list[str]) -> set[str]:
        return {h for h in hashes if h in self.known_hashes}

    async def store_chunks(self, chunks, embeddings, conversation_id=None) -> int:
        self.stored_chunks.extend(chunks)
        return len(chunks)

    async def fts_candidates(self, query, k):
        return []

    async def vector_candidates(self, query_embedding, k, conversation_id=None):
        return []


async def test_embed_skip_skips_already_stored_chunks():
    from src.core.tools.chunking import chunk_markdown_data

    embedder = RecordingEmbedder()
    # Compute the real chunk pieces for page a exactly like the pipeline,
    # then report their hashes as already stored.
    page_a_chunks = chunk_markdown_data(
        [{"url": "http://example.com/a", "content": sample_markdown()}]
    )
    known = {
        chunk_hash(
            (c.get("metadata") or {}).get("url", "") or c.get("url", ""),
            c["content"],
        )
        for c in page_a_chunks
    }
    repo = PreStoredChunkRepo(known_hashes=known)
    toolbox = AskToolbox(
        FakeSearchProvider(),
        make_scraper(),
        FakeRanker(),
        embedder=embedder,
        chunk_repo=repo,
    )
    await toolbox.web_search("rust")

    await toolbox.fetch_pages()

    # Page a is already stored: only page b's chunks get embedded.
    assert len(embedder.embedded) == 2
    assert repo.stored_chunks and all(
        (c.get("metadata") or {}).get("url") == "http://example.com/b"
        for c in repo.stored_chunks
    )


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


# <----- Retrieval scope (WU-5) and recency (WU-11) ----->


class RecordingVectorRepo:
    """Vector-store fake that records the scoping arguments."""

    def __init__(self):
        self.stored_conversation_id = "unset"
        self.query_conversation_id = "unset"

    async def store_chunks(self, chunks, embeddings, conversation_id=None):
        self.stored_conversation_id = conversation_id
        return len(chunks)

    async def existing_hashes(self, hashes):
        return set()

    async def vector_candidates(self, query_embedding, k, conversation_id=None):
        self.query_conversation_id = conversation_id
        return []

    async def fts_candidates(self, query, k):
        return []


async def test_scope_request_skips_vector_retrieval():
    repo = RecordingVectorRepo()
    embedder = RecordingEmbedder()
    toolbox = AskToolbox(
        FakeSearchProvider(),
        make_scraper(),
        FakeRanker(),
        embedder=embedder,
        chunk_repo=repo,
        scope="request",
        conversation_id=uuid.uuid4(),
    )
    await toolbox.web_search("rust")
    await toolbox.fetch_pages()
    await toolbox.get_context("rust")

    assert toolbox.used_vector_retrieval is False
    assert repo.query_conversation_id == "unset"  # never queried
    # But chunks are still embedded + scoped when stored.
    assert repo.stored_conversation_id is not None


async def test_scope_conversation_passes_conversation_id_to_candidates():
    conversation_id = uuid.uuid4()
    repo = RecordingVectorRepo()
    toolbox = AskToolbox(
        FakeSearchProvider(),
        make_scraper(),
        FakeRanker(),
        embedder=RecordingEmbedder(),
        chunk_repo=repo,
        scope="conversation",
        conversation_id=conversation_id,
    )
    await toolbox.web_search("rust")
    await toolbox.fetch_pages()
    await toolbox.get_context("rust")

    assert toolbox.used_vector_retrieval is True
    assert repo.query_conversation_id == conversation_id


async def test_scope_global_never_filters():
    repo = RecordingVectorRepo()
    toolbox = AskToolbox(
        FakeSearchProvider(),
        make_scraper(),
        FakeRanker(),
        embedder=RecordingEmbedder(),
        chunk_repo=repo,
        scope="global",
        conversation_id=uuid.uuid4(),
    )
    await toolbox.web_search("rust")
    await toolbox.fetch_pages()
    await toolbox.get_context("rust")

    assert repo.query_conversation_id is None


async def test_recency_bias_reorders_selected_chunks():
    now = datetime(2026, 10, 26, tzinfo=UTC)

    def make_pool():
        return [
            {
                "content": "old and strong",
                "metadata": {
                    "relevance_score": 0.9,
                    "published_at": "2023-01-01T00:00:00+00:00",
                },
            },
            {
                "content": "fresh and weak",
                "metadata": {
                    "relevance_score": 0.95,
                    "published_at": "2026-10-20T00:00:00+00:00",
                },
            },
        ]

    toolbox = AskToolbox(
        FakeSearchProvider(), make_scraper(), FakeRanker(), recency_bias=1.0, now=now
    )
    toolbox.chunks = make_pool()

    await toolbox.get_context("q")

    # fresh: 0.95 * 2.0 = 1.9 beats old: 0.9 * 1.3 = 1.17
    assert toolbox.selected_chunks[0]["content"] == "fresh and weak"

    neutral = AskToolbox(
        FakeSearchProvider(), make_scraper(), FakeRanker()
    )
    neutral.chunks = make_pool()
    await neutral.get_context("q")
    assert neutral.selected_chunks[0]["content"] == "old and strong"


async def test_recency_bias_clamped_to_ceiling():
    toolbox = AskToolbox(
        FakeSearchProvider(), make_scraper(), FakeRanker(), recency_bias=99.0
    )
    assert toolbox.recency_bias == settings.RECENCY_BIAS_MAX


# <----- Hybrid retrieval blend (WU-4) ----->


class HybridVectorRepo(RecordingVectorRepo):
    """Vector leg returns both chunks; lexical leg returns one."""

    def __init__(self):
        super().__init__()
        self.fts_calls = 0
        self.vector: list[dict] = []
        self.lexical: list[dict] = []

    def load(self, vector: list[dict], lexical: list[dict]):
        self.vector = vector
        self.lexical = lexical
        return self

    async def fts_candidates(self, query, k):
        self.fts_calls += 1
        return self.lexical

    async def vector_candidates(self, query_embedding, k, conversation_id=None):
        return self.vector


async def test_hybrid_default_alpha_never_queries_the_lexical_leg():
    repo = HybridVectorRepo().load(
        [{"content": "vector chunk", "metadata": {}}], []
    )
    toolbox = AskToolbox(
        FakeSearchProvider(), make_scraper(), FakeRanker(),
        embedder=RecordingEmbedder(), chunk_repo=repo,
    )
    await toolbox.web_search("rust")
    await toolbox.fetch_pages()
    await toolbox.get_context("rust")

    assert repo.fts_calls == 0
    assert toolbox.used_vector_retrieval is True


async def test_hybrid_fuses_lexical_leg_into_the_pool():
    repo = HybridVectorRepo().load(
        [
            {"content": "vector-only chunk about postgres btree", "metadata": {}},
            {"content": "shared chunk", "metadata": {}},
        ],
        [
            {"content": "shared chunk", "metadata": {}},
            {"content": "lexical-only chunk", "metadata": {}},
        ],
    )
    toolbox = AskToolbox(
        FakeSearchProvider(), make_scraper(), FakeRanker(),
        embedder=RecordingEmbedder(), chunk_repo=repo, hybrid_alpha=0.5,
    )
    await toolbox.web_search("rust")
    await toolbox.fetch_pages()
    toolbox.chunks = []  # ensure the pool came from the fused legs
    await toolbox.get_context("rust")

    assert repo.fts_calls == 1
    contents = [c["content"] for c in toolbox.selected_chunks]
    # Union of both legs survives; the double-leg candidate leads.
    assert "shared chunk" in contents
    assert "lexical-only chunk" in contents
    assert contents[0] == "shared chunk"


async def test_hybrid_lexical_failure_degrades_to_vector(monkeypatch):
    repo = HybridVectorRepo().load(
        [{"content": "vector chunk", "metadata": {}}], []
    )

    async def exploding_fts(query, k):
        raise RuntimeError("fts down")

    repo.fts_candidates = exploding_fts
    toolbox = AskToolbox(
        FakeSearchProvider(), make_scraper(), FakeRanker(),
        embedder=RecordingEmbedder(), chunk_repo=repo, hybrid_alpha=0.5,
    )
    await toolbox.web_search("rust")
    await toolbox.fetch_pages()
    context = await toolbox.get_context("rust")

    assert toolbox.used_vector_retrieval is True
    assert "vector chunk" in context


async def test_hybrid_both_legs_fail_falls_back_in_memory():
    class BothExplodeRepo(HybridVectorRepo):
        async def fts_candidates(self, query, k):
            raise RuntimeError("fts down")

        async def vector_candidates(self, query_embedding, k, conversation_id=None):
            raise RuntimeError("pgvector down")

    toolbox = AskToolbox(
        FakeSearchProvider(), make_scraper(), FakeRanker(),
        embedder=RecordingEmbedder(), chunk_repo=BothExplodeRepo(),
        hybrid_alpha=0.5,
    )
    await toolbox.web_search("rust")
    await toolbox.fetch_pages()
    context = await toolbox.get_context("rust")

    assert toolbox.used_vector_retrieval is False
    assert "first paragraph" in context
