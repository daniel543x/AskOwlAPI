"""Offline tests for alpha-weighted RRF fusion (WU-4)."""

from src.core.tools.rrf import rrf_fuse


def _candidate(content: str) -> dict:
    return {"content": content, "url": f"https://a.dev/{content}"}


VECTOR = [_candidate("only vector"), _candidate("both legs"), _candidate("vector two")]
LEXICAL = [_candidate("both legs"), _candidate("only lexical")]


def test_both_leg_candidates_accumulate_mass():
    fused = rrf_fuse(VECTOR, LEXICAL, alpha=0.5)

    # "both legs" ranks first in both legs: its fused mass wins.
    assert fused[0]["content"] == "both legs"
    # Every candidate survives the fusion (union of both pools).
    contents = {c["content"] for c in fused}
    assert contents == {"only vector", "both legs", "vector two", "only lexical"}


def test_alpha_one_is_pure_vector_order():
    fused = rrf_fuse(VECTOR, LEXICAL, alpha=1.0)

    assert [c["content"] for c in fused] == [
        "only vector",
        "both legs",
        "vector two",
    ]


def test_alpha_zero_is_pure_lexical_order():
    fused = rrf_fuse(VECTOR, LEXICAL, alpha=0.0)

    assert [c["content"] for c in fused] == ["both legs", "only lexical"]


def test_alpha_extremes_clamped():
    fused = rrf_fuse(VECTOR, LEXICAL, alpha=5.0)
    assert [c["content"] for c in fused] == [
        "only vector",
        "both legs",
        "vector two",
    ]
    fused = rrf_fuse(VECTOR, LEXICAL, alpha=-1.0)
    assert [c["content"] for c in fused] == ["both legs", "only lexical"]


def test_single_leg_degenerates_to_that_leg():
    fused = rrf_fuse(VECTOR, [], alpha=0.5)
    assert [c["content"] for c in fused] == [
        "only vector",
        "both legs",
        "vector two",
    ]

    fused = rrf_fuse([], LEXICAL, alpha=0.5)
    assert [c["content"] for c in fused] == ["both legs", "only lexical"]


def test_empty_legs_give_empty_pool():
    assert rrf_fuse([], [], alpha=0.5) == []
