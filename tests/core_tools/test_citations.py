"""Offline tests for deterministic citation validation (WU-9 / AR-5)."""

from src.core.tools.citations import validate_citations

CHUNKS = [
    {"n": 1, "url": "https://a.dev/x", "title": "A"},
    {"n": 2, "url": "https://b.dev/y", "title": "B"},
]


def test_valid_refs_resolve_with_full_coverage():
    result = validate_citations("Claim one [1] and claim two [2].", CHUNKS)
    assert result == {"cited": [1, 2], "unknown": [], "coverage": 1.0}


def test_unknown_refs_are_surfaced_never_rewritten():
    result = validate_citations("As shown in [7] and [2].", CHUNKS)
    assert result["cited"] == [2]
    assert result["unknown"] == [7]
    assert result["coverage"] == 0.5


def test_duplicates_count_once():
    result = validate_citations("[1] again [1] and [1].", CHUNKS)
    assert result["cited"] == [1]
    assert result["coverage"] == 0.5


def test_legacy_markdown_links_do_not_count_as_refs():
    result = validate_citations("Claim [Source](https://a.dev/x).", CHUNKS)
    assert result["cited"] == []
    assert result["coverage"] == 0.0


def test_escaped_brackets_are_ignored():
    result = validate_citations("Use \\[1\\] literally, but cite [1].", CHUNKS)
    assert result["cited"] == [1]
    assert result["unknown"] == []


def test_no_chunks_means_zero_coverage():
    result = validate_citations("Anything [1].", [])
    assert result == {"cited": [], "unknown": [1], "coverage": 0.0}


def test_no_refs_at_all():
    result = validate_citations("A claim without citations.", CHUNKS)
    assert result == {"cited": [], "unknown": [], "coverage": 0.0}
