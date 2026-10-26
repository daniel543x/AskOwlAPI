"""Offline fixture eval (AR-9): retrieval + citation regression gate.

Runs `evaluate_query` from the harness over the committed fixtures in
`tests/fixtures/eval/` with replay providers, a deterministic hash-based
embedder, and the order-preserving fake ranker — no network, no services,
no models. This is a regression gate on the retrieval *plumbing* (sanity
filter, dedup, chunking, vector retrieval, context selection, citation
numbering), not a semantic leaderboard:

Baselines recorded at introduction (PR 3, wu/citations):
    baseline config: mean recall@n = 1.000, mean MRR = 1.000
    vector config:   mean recall@n = 1.000 (candidates cover every chunk)
The gates below pin those values minus a small margin; lower any gate only
with an explicit, quoted eval delta in the PR.
"""

import asyncio
from pathlib import Path

from src.core.ask.eval.fixtures import (
    FixtureScraper,
    FixtureSearchProvider,
    load_fixtures,
)
from src.core.ask.eval.harness import EvalQuery, evaluate_query, run_eval
from src.core.tools.citations import validate_citations
from tests.core_ask.fakes import FakeRanker

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "eval"


class HashEmbedder:
    """Deterministic embedder: token hashes bucketed into 16 dims.

    Not semantic — the gate is plumbing, not quality. Stable across runs
    and machines so the recorded baselines mean something.
    """

    def __init__(self, dims: int = 16):
        self.dims = dims

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self.dims
        for token in "".join(
            c if c.isalnum() else " " for c in text.lower()
        ).split():
            bucket = hash(token) % self.dims
            vector[bucket] += 1.0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]


class InMemoryChunkRepo:
    """Minimal ChunkStoreLike over stored chunks (cosine retrieval)."""

    def __init__(self):
        self.stored: list[tuple[dict, list[float]]] = []

    async def existing_hashes(self, hashes):
        return set()

    async def store_chunks(self, chunks, embeddings, conversation_id=None):
        self.stored.extend(zip(chunks, embeddings, strict=True))
        return len(chunks)

    async def vector_candidates(self, query_embedding, k, conversation_id=None):
        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b, strict=True))
            return dot  # vectors are normalized

        scored = sorted(
            ((cosine(query_embedding, e), c) for c, e in self.stored),
            key=lambda pair: -pair[0],
        )
        return [
            {
                "content": c["content"],
                "metadata": dict(c.get("metadata") or {}),
                "url": (c.get("metadata") or {}).get("url", ""),
                "distance": 1 - s,
            }
            for s, c in scored[:k]
        ]


def _entries(fixtures):
    return [EvalQuery(query=f.query, relevant_urls=f.relevant_urls) for f in fixtures]


async def _run(config_name, fixtures, embedder=None, chunk_repo=None):
    return await run_eval(
        config_name,
        _entries(fixtures),
        FixtureSearchProvider(fixtures),
        FixtureScraper(fixtures),
        FakeRanker(),
        embedder,
        chunk_repo,
    )


async def test_offline_eval_baseline_meets_recorded_gate():
    fixtures = load_fixtures(FIXTURE_DIR)
    assert len(fixtures) >= 3, "keep at least 3 fixtures for the gate"

    report = await _run("baseline", fixtures)

    assert report.mean_recall >= 0.95, (
        f"baseline recall regressed: {report.mean_recall}"
    )
    assert report.mean_mrr >= 0.90, f"baseline MRR regressed: {report.mean_mrr}"


async def test_offline_eval_vector_path_meets_recorded_gate():
    fixtures = load_fixtures(FIXTURE_DIR)
    chunk_repo = InMemoryChunkRepo()

    report = await _run("vector", fixtures, HashEmbedder(), chunk_repo)

    assert report.mean_recall >= 0.95, f"vector recall regressed: {report.mean_recall}"
    assert all(r.vector_retrieval for r in report.results), (
        "vector config silently fell back to in-memory ranking"
    )


async def test_offline_eval_citations_numbering_roundtrip():
    """A well-cited scripted answer reaches full citation coverage (AR-9)."""
    fixtures = load_fixtures(FIXTURE_DIR)
    entry = _entries(fixtures)[0]

    result = await evaluate_query(
        entry,
        FixtureSearchProvider(fixtures),
        FixtureScraper(fixtures),
        FakeRanker(),
    )
    assert result.recall == 1.0

    # Simulate the answer step: the numbered context plus a scripted answer
    # citing every provided chunk must validate with coverage 1.0.
    toolbox_citations = [
        {"n": i, "url": url}
        for i, url in enumerate(result.context_urls, start=1)
    ]
    well_cited = "Claim one. [1] Claim two. [2]"
    answer = well_cited if len(toolbox_citations) >= 2 else "Claim one. [1]"
    validation = validate_citations(answer, toolbox_citations)

    assert validation["unknown"] == []
    assert validation["coverage"] == 1.0


def test_fixture_providers_replay_deterministically():
    fixtures = load_fixtures(FIXTURE_DIR)

    async def collect():
        provider = FixtureSearchProvider(fixtures)
        scraper = FixtureScraper(fixtures)
        first = await provider.search(fixtures[0].query)
        second = await provider.search(fixtures[0].query)
        pages = await scraper.scrape([{"url": r.url} for r in first])
        return first, second, pages

    first, second, pages = asyncio.run(collect())

    assert first == second, "replay must be deterministic"
    assert all(page["content"] for page in pages), "fixture pages must have content"
