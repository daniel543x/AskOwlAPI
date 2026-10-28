"""Offline tests for the recency boost (WU-11 / AR-7)."""

from datetime import UTC, datetime, timedelta

import pytest

from src.core.tools.recency import apply_recency_bias, recency_weight

NOW = datetime(2026, 10, 26, tzinfo=UTC)


def _iso(days_ago: float) -> str:
    return (NOW - timedelta(days=days_ago)).isoformat()


def test_recency_weight_tiers():
    assert recency_weight(_iso(10), NOW) == 1.0  # < 30 days
    assert recency_weight(_iso(200), NOW) == 0.6  # < 1 year
    assert recency_weight(_iso(1000), NOW) == 0.3  # < 3 years
    assert recency_weight(_iso(2000), NOW) == 0.0  # older
    assert recency_weight(None, NOW) == 0.0  # unknown: never boosted
    assert recency_weight("not-a-date", NOW) == 0.0


def test_future_dates_do_not_explode():
    assert recency_weight(_iso(-5), NOW) == 1.0  # clamped to the top tier


def test_apply_recency_bias_reorders_by_adjusted_score():
    chunks = [
        {
            "content": "old and strongly scored",
            "metadata": {
                "relevance_score": 0.9,
                "published_at": _iso(1000),
            },
        },
        {
            "content": "fresh but weakly scored",
            "metadata": {
                "relevance_score": 0.5,
                "published_at": _iso(5),
            },
        },
    ]

    reordered = apply_recency_bias(chunks, bias=1.0, now=NOW)

    # old: 0.9 * (1 + 1*0.3) = 1.17; fresh: 0.5 * (1 + 1*1.0) = 1.0 — the
    # old chunk still wins; scores are adjusted in place.
    assert reordered[0]["metadata"]["relevance_score"] == pytest.approx(1.17)
    assert reordered[1]["metadata"]["relevance_score"] == pytest.approx(1.0)

    chunks[1]["metadata"]["relevance_score"] = 0.95  # fresh: 1.9 > 1.17
    reordered = apply_recency_bias(chunks, bias=1.0, now=NOW)
    assert reordered[0]["content"] == "fresh but weakly scored"


def test_apply_recency_bias_zero_is_noop():
    chunks = [{"content": "a", "metadata": {"relevance_score": 0.5}}]

    result = apply_recency_bias(chunks, bias=0.0, now=NOW)

    assert result == chunks
    assert chunks[0]["metadata"]["relevance_score"] == 0.5


def test_apply_recency_bias_missing_scores_keep_order():
    chunks = [{"content": "a", "metadata": {}}, {"content": "b", "metadata": {}}]

    result = apply_recency_bias(chunks, bias=1.0, now=NOW)

    assert [c["content"] for c in result] == ["a", "b"]
