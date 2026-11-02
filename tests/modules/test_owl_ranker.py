"""OwlRanker batching: FlashRank must never receive the whole pool as one
ONNX batch. Single-call reranking of ~1200 passages allocates O(batch x
seq^2) attention memory in onnxruntime (measured: a 3 GiB buffer for the
first attention node alone, ~22 GB process RSS) and was the root cause of
the pipeline OOM (ref/OOM_INVESTIGATION.md). These tests pin the batching
contract: fixed-size batches, globally merged ordering, and the existing
failure fallback."""

import pytest

from src.modules.owl_reranker.owl_reranker import OwlRanker
from src.modules.search_engine.providers.base import SearchResult
from src.tools.settings import settings


class FakeFlashRank:
    def __init__(self, model_name=None, cache_dir=None, fail_after=None, best_id="77"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.fail_after = fail_after
        self.best_id = best_id
        self.batch_sizes = []
        self.seen_ids = []

    def rerank(self, request):
        self.batch_sizes.append(len(request.passages))
        if self.fail_after is not None and len(self.batch_sizes) > self.fail_after:
            raise RuntimeError("onnx arena exploded")
        for passage in request.passages:
            self.seen_ids.append(passage["id"])
            score = (
                100.0
                if passage["id"] == self.best_id
                else float(int(passage["id"]) % 5)
            )
            passage["score"] = score
        request.passages.sort(key=lambda p: p["score"], reverse=True)
        return request.passages


@pytest.fixture
def make_ranker(monkeypatch):
    def _make(**fake_kwargs):
        fake = FakeFlashRank(**fake_kwargs)

        def factory(model_name=None, cache_dir=None):
            fake.model_name = model_name
            fake.cache_dir = cache_dir
            return fake

        monkeypatch.setattr(
            "src.modules.owl_reranker.owl_reranker.Ranker", factory
        )
        ranker = OwlRanker(model_name="m", cache_dir="/tmp/x")
        return ranker, fake

    return _make


def test_rank_chunks_reranks_in_fixed_batches(make_ranker, monkeypatch):
    monkeypatch.setattr(settings, "RANKER_BATCH_SIZE", 32)
    ranker, fake = make_ranker()
    chunks = [{"content": f"chunk {i}"} for i in range(100)]

    ranker.rank_chunks("q", chunks, top_n=20)

    assert fake.batch_sizes == [32, 32, 32, 4]
    assert len(fake.seen_ids) == 100
    assert fake.seen_ids[0] == "0"
    assert fake.seen_ids[-1] == "99"


def test_rank_chunks_merges_scores_across_batches(make_ranker, monkeypatch):
    monkeypatch.setattr(settings, "RANKER_BATCH_SIZE", 32)
    ranker, _ = make_ranker()
    chunks = [{"content": f"chunk {i}"} for i in range(100)]

    ranked = ranker.rank_chunks("q", chunks, top_n=5)

    assert ranked[0]["content"] == "chunk 77"
    assert ranked[0]["metadata"]["relevance_score"] == 100.0
    assert len(ranked) == 5
    scores = [c["metadata"]["relevance_score"] for c in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_chunks_single_batch_when_pool_fits(make_ranker, monkeypatch):
    monkeypatch.setattr(settings, "RANKER_BATCH_SIZE", 32)
    ranker, fake = make_ranker()
    chunks = [{"content": f"chunk {i}"} for i in range(10)]

    ranked = ranker.rank_chunks("q", chunks, top_n=3)

    assert fake.batch_sizes == [10]
    assert len(ranked) == 3


def test_rank_chunks_falls_back_when_a_batch_fails(make_ranker, monkeypatch):
    monkeypatch.setattr(settings, "RANKER_BATCH_SIZE", 32)
    ranker, _ = make_ranker(fail_after=1)
    chunks = [{"content": f"chunk {i}"} for i in range(100)]

    ranked = ranker.rank_chunks("q", chunks, top_n=20)

    assert len(ranked) == 20
    assert ranked[0]["content"] == "chunk 0"
    assert "metadata" not in ranked[0] or "relevance_score" not in ranked[0][
        "metadata"
    ]


def test_rank_chunks_empty_input(make_ranker):
    ranker, _ = make_ranker()
    assert ranker.rank_chunks("q", [], top_n=5) == []
    assert ranker.rank_chunks("q", None, top_n=5) == []


def test_rank_web_search_reranks_in_fixed_batches(make_ranker, monkeypatch):
    monkeypatch.setattr(settings, "RANKER_BATCH_SIZE", 16)
    ranker, fake = make_ranker(best_id="35")
    results = [
        SearchResult(title=f"t{i}", snippet=f"s{i}", url=f"https://x/{i}")
        for i in range(40)
    ]

    ranked = ranker.rank_web_search("q", results, top_n=10)

    assert fake.batch_sizes == [16, 16, 8]
    assert ranked[0]["url"] == "https://x/35"
    assert len(ranked) == 10
    scores = [r["score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)
