import math

from src.core.ask.eval.harness import (
    EvalQuery,
    _context_urls,
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
