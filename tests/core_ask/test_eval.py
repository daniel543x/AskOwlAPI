import math

from src.core.ask.eval.harness import (
    CachedScraper,
    CachedSearchProvider,
    EvalQuery,
    EvalReport,
    QueryResult,
    _context_urls,
    comparison_verdict,
    evaluate_query,
    load_golden_set,
    run_eval,
)
from src.core.ask.eval.metrics import mrr, recall_at_k
from tests.core_ask.fakes import (
    FakeRanker,
    FakeScraper,
    FakeSearchProvider,
    make_scraper,
)

# <----- metrics ----->


def test_recall_at_k_basic():
    retrieved = ["a", "b", "c", "d"]
    assert recall_at_k(retrieved, {"a", "c"}, k=4) == 1.0
    assert recall_at_k(retrieved, {"a", "z"}, k=4) == 0.5
    assert recall_at_k(retrieved, {"a"}, k=1) == 1.0
    assert recall_at_k(retrieved, {"b"}, k=1) == 0.0


def test_recall_at_k_edge_cases():
    assert recall_at_k([], {"a"}, k=5) == 0.0
    assert recall_at_k(["a"], set(), k=5) == 0.0


def test_mrr_basic():
    assert mrr(["x", "a", "y"], {"a"}) == 0.5
    assert mrr(["a"], {"a"}) == 1.0
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_metrics_ignore_url_variants():
    retrieved = ["https://rust-lang.org/learn/", "https://OTHER.example/docs"]
    relevant = {"https://www.rust-lang.org/learn"}
    assert recall_at_k(retrieved, relevant, k=2) == 1.0
    assert mrr(retrieved, relevant) == 1.0


def test_metrics_keep_scheme_distinct():
    assert recall_at_k(["http://example.com/a"], {"https://example.com/a"}, k=1) == 0.0


# <----- harness helpers ----->


def test_context_urls_deduplicates_and_orders():
    chunks = [
        {"metadata": {"url": "http://a"}},
        {"metadata": {"url": "http://b"}},
        {"url": "http://a"},
        {"metadata": {}},
    ]
    assert _context_urls(chunks) == ["http://a", "http://b"]


def test_golden_set_loads_and_is_sized():
    entries = load_golden_set()
    assert len(entries) >= 20
    assert all(entry.query and entry.relevant_urls for entry in entries)


# <----- caching wrappers ----->


class CountingSearch(FakeSearchProvider):
    def __init__(self):
        super().__init__()
        self.calls = 0

    async def search(self, query, engines=None):
        self.calls += 1
        return await super().search(query)


async def test_cached_search_provider_hits_inner_once_per_query():
    inner = CountingSearch()
    cached = CachedSearchProvider(inner)

    first = await cached.search(query="rust")
    second = await cached.search(query="rust")
    await cached.search(query="python")

    assert inner.calls == 2
    assert cached.query_count == 2
    assert [r.url for r in first] == [r.url for r in second]
    assert first is not second, "callers get a copy, not the cached list"


async def test_cached_scraper_fetches_each_url_once_and_preserves_order():
    inner = FakeScraper(pages={"http://a": "content a", "http://b": "content b"})
    cached = CachedScraper(inner)

    first = await cached.scrape([{"url": "http://a"}, {"url": "http://b"}])
    second = await cached.scrape([{"url": "http://b"}, {"url": "http://a"}])

    assert inner.requested == ["http://a", "http://b"], "each URL fetched once"
    assert cached.fetch_count == 2
    assert [p["url"] for p in first] == ["http://a", "http://b"]
    assert [p["url"] for p in second] == ["http://b", "http://a"]
    assert first[0] is not second[1], "callers get copies, not shared dicts"


async def test_cached_scraper_retries_failed_fetches():
    inner = FakeScraper(pages={"http://a": ""})
    cached = CachedScraper(inner)

    first = await cached.scrape([{"url": "http://a"}])
    inner.pages["http://a"] = "recovered content"
    second = await cached.scrape([{"url": "http://a"}])

    assert first[0]["content"] == ""
    assert second[0]["content"] == "recovered content"
    assert inner.requested == ["http://a", "http://a"], "failure not cached"


# <----- end-to-end with fakes ----->


async def test_evaluate_query_baseline_scores_relevant_sources():
    entry = EvalQuery(
        query="rust",
        relevant_urls=["http://example.com/a"],
    )

    result = await evaluate_query(
        entry, FakeSearchProvider(), make_scraper(), FakeRanker()
    )

    assert result.context_urls == ["http://example.com/a", "http://example.com/b"]
    assert result.recall == 1.0
    assert result.reciprocal_rank == 1.0
    assert result.vector_retrieval is False
    assert result.latency_ms >= 0


class FakeEmbedder:
    """Deterministic 2-D embedder: one basis vector per source URL."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    @property
    def dims(self) -> int:
        return 2

    async def embed_texts(self, texts: list[str]):
        return [self._vector_for(text) for text in texts]

    def _vector_for(self, text: str) -> list[float]:
        for key, vector in self.vectors.items():
            if key in text:
                return vector
        return [0.0, 0.0]


class FakeChunkRepo:
    """In-memory chunk store with cosine-similarity retrieval."""

    def __init__(self):
        self.stored: list[tuple[dict, list[float]]] = []

    async def store_chunks(self, chunks, embeddings):
        self.stored.extend(zip(chunks, embeddings, strict=True))
        return len(chunks)

    async def vector_candidates(self, query_embedding, k):
        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b, strict=True))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        scored = [
            (cosine(query_embedding, embedding), chunk)
            for chunk, embedding in self.stored
        ]
        scored.sort(key=lambda pair: -pair[0])
        return [
            {
                "content": chunk["content"],
                "metadata": dict(chunk.get("metadata") or {}),
                "url": (chunk.get("metadata") or {}).get("url", ""),
                "distance": 1 - score,
            }
            for score, chunk in scored[:k]
        ]


async def test_evaluate_query_vector_path_uses_store():
    entry = EvalQuery(
        query="rust ownership",
        relevant_urls=["http://example.com/a"],
    )
    body_a = "# Rust\nownership content about rust language"
    body_b = "# Other\nunrelated cooking recipes"

    scraper = FakeScraper(
        pages={"http://example.com/a": body_a, "http://example.com/b": body_b}
    )
    embedder = FakeEmbedder(
        {
            "http://example.com/a": [1.0, 0.0],
            "cooking": [0.0, 1.0],
        }
    )
    chunk_repo = FakeChunkRepo()

    from src.modules.owl_reranker.base import IRanker

    class KeepOrderRanker(IRanker):
        def rank_web_search(self, query, search_data, top_n=None):
            ranked = [{"url": r.url, "score": 1.0} for r in search_data]
            return ranked[:top_n] if top_n else ranked

        def rank_chunks(self, query, chunks=None, top_n=None):
            chunks = chunks or []
            return chunks[:top_n] if top_n else chunks

    result = await evaluate_query(
        entry,
        FakeSearchProvider(),
        scraper,
        KeepOrderRanker(),
        embedder=embedder,
        chunk_repo=chunk_repo,
    )

    assert result.vector_retrieval is True
    assert chunk_repo.stored, "chunks were embedded and stored"
    # Query vector aligns with page A chunks; they come first.
    assert result.context_urls[0] == "http://example.com/a"


async def test_run_eval_aggregates():
    entries = [
        EvalQuery(query="rust", relevant_urls=["http://example.com/a"]),
        EvalQuery(query="other", relevant_urls=["http://nowhere.example"]),
    ]

    report = await run_eval(
        "test-config",
        entries,
        FakeSearchProvider(),
        make_scraper(),
        FakeRanker(),
    )

    assert report.config_name == "test-config"
    assert len(report.results) == 2
    assert report.mean_recall == 0.5
    assert report.to_dict()["config_name"] == "test-config"


# <----- comparison verdict ----->


def _make_report(recalls, vector_flags, name="cfg"):
    results = [
        QueryResult(
            query=f"q{i}",
            context_urls=[],
            relevant_urls=[],
            recall=recall,
            reciprocal_rank=0.0,
            latency_ms=0.0,
            vector_retrieval=flag,
        )
        for i, (recall, flag) in enumerate(
            zip(recalls, vector_flags, strict=True)
        )
    ]
    return EvalReport(
        config_name=name,
        results=results,
        mean_recall=sum(recalls) / (len(recalls) or 1),
        mean_mrr=0.0,
        mean_latency_ms=0.0,
    )


def test_comparison_verdict_fails_when_vector_degraded():
    baseline = _make_report([1.0, 1.0], [False, False], name="baseline")
    vector = _make_report([1.0, 1.0], [True, False], name="vector")

    passed, detail = comparison_verdict(baseline, vector)

    assert passed is False
    assert "fell back" in detail
    assert "1/2" in detail


def test_comparison_verdict_passes_on_clean_tie_or_win():
    baseline = _make_report([0.5, 0.0], [False, False], name="baseline")
    clean_tie = _make_report([0.5, 0.0], [True, True], name="vector")
    win = _make_report([1.0, 0.5], [True, True], name="vector")

    assert comparison_verdict(baseline, clean_tie)[0] is True
    assert comparison_verdict(baseline, win)[0] is True


def test_comparison_verdict_fails_on_clean_loss():
    baseline = _make_report([1.0], [False], name="baseline")
    vector = _make_report([0.0], [True], name="vector")

    passed, detail = comparison_verdict(baseline, vector)

    assert passed is False
    assert "recall 0.000 vs 1.000" in detail
