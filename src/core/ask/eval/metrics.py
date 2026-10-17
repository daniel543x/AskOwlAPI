"""Retrieval-quality metrics for the eval harness (pure functions)."""

from collections.abc import Iterable, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant URLs present in the top-k retrieved URLs."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = list(retrieved)[:k]
    hits = sum(1 for url in top_k if url in relevant_set)
    return hits / len(relevant_set)


def mrr(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Mean reciprocal rank (single query): 1 / rank of first relevant URL."""
    relevant_set = set(relevant)
    for rank, url in enumerate(retrieved, start=1):
        if url in relevant_set:
            return 1.0 / rank
    return 0.0
