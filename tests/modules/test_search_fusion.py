"""Offline tests for search-result dedup and fusion scoring (AR-2 / WU-3)."""

from src.modules.search_engine.fusion import (
    MergedResult,
    apply_fusion_floor,
    authority_for,
    merge_results,
)
from src.modules.search_engine.providers.base import SearchResult

AUTHORITY = {".gov": 0.9, "github.com": 0.8}


def test_same_url_from_two_engines_merges_once():
    results = [
        SearchResult("Short title", "snippet one", "https://example.com/a"),
        SearchResult(
            "A much longer surviving title",
            "a much longer surviving snippet",
            "https://www.example.com/a?utm_source=x",
        ),
    ]

    merged = merge_results(results, AUTHORITY)

    assert len(merged) == 1
    entry = merged[0]
    assert entry.url == "https://example.com/a"
    assert entry.merged_count == 2
    assert entry.title == "A much longer surviving title"
    assert entry.snippet == "a much longer surviving snippet"
    assert entry.original_rank == 0
    # Engine agreement term: min(2/3, 1) > single-engine results.
    assert entry.fusion_score > 0


def test_first_appearance_order_preserved():
    results = [
        SearchResult("b", "s", "https://example.com/b"),
        SearchResult("a", "s", "https://example.com/a"),
    ]

    merged = merge_results(results, AUTHORITY)

    assert [m.url for m in merged] == ["https://example.com/b", "https://example.com/a"]


def test_fusion_formula():
    merged = merge_results(
        [
            SearchResult("t", "s", "https://github.com/x"),
            SearchResult("t2", "s2", "https://unknown.example/y"),
        ],
        AUTHORITY,
    )
    github, unknown = merged

    # Position term for the first result is 1.0; authority github.com = 0.8.
    assert github.fusion_score == 0.4 * (1 / 3) + 0.3 * 0.8 + 0.3 * 1.0
    # Unknown hosts score 0.5 authority; second result position = 1/1.1.
    assert unknown.fusion_score == 0.4 * (1 / 3) + 0.3 * 0.5 + 0.3 * (1 / 1.1)


def test_authority_matching():
    assert authority_for("docs.python.gov", AUTHORITY) == 0.9
    assert authority_for("gist.github.com", AUTHORITY) == 0.8
    assert authority_for("github.com.evil.example", AUTHORITY) == 0.5
    assert authority_for("unknown.example", AUTHORITY) == 0.5


def test_as_search_result_roundtrip():
    entry = MergedResult(
        url="https://example.com/a",
        title="t",
        snippet="s",
        merged_count=1,
        original_rank=0,
        fusion_score=0.5,
    )
    result = entry.as_search_result()
    assert (result.title, result.snippet, result.url) == ("t", "s", "https://example.com/a")


def test_fusion_floor_off_when_zero():
    merged = merge_results(
        [SearchResult("t", "s", "https://example.com/a")], AUTHORITY
    )
    kept, dropped = apply_fusion_floor(merged, 0.0)
    assert kept == merged and dropped == 0


def test_fusion_floor_drops_below_and_never_empties():
    results = [
        SearchResult("t1", "s1", "https://example.com/1"),
        SearchResult("t2", "s2", "https://example.com/2"),
        SearchResult("t3", "s3", "https://example.com/3"),
        SearchResult("t4", "s4", "https://example.com/4"),
        SearchResult("t5", "s5", "https://example.com/5"),
    ]
    merged = merge_results(results, AUTHORITY)

    kept, dropped = apply_fusion_floor(merged, 1.0)  # unreachable floor

    # Degrade-don't-empty: the three best-scoring drops are rescued.
    assert len(kept) == 3
    assert dropped == 2
    kept_scores = [m.fusion_score for m in kept]
    assert kept_scores == sorted(kept_scores, reverse=True)


def test_fusion_floor_rescues_in_original_order():
    results = [
        SearchResult("low", "s", "https://unknown.example/low"),  # rank 0, low fusion
        SearchResult("t2", "s", "https://github.com/x"),  # authority boost
        SearchResult("t3", "s", "https://unknown.example/y"),
        SearchResult("t4", "s", "https://unknown.example/z"),
    ]
    merged = merge_results(results, AUTHORITY)

    kept, _ = apply_fusion_floor(merged, 1.0)

    assert len(kept) == 3
    # Original relative order is preserved among the rescued set.
    urls = [m.url for m in merged]
    assert [m.url for m in kept] == [u for u in urls if u in {m.url for m in kept}]
