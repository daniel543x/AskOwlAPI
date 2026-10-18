"""Retrieval-quality metrics for the eval harness (pure functions)."""

from collections.abc import Iterable, Sequence
from urllib.parse import urlsplit


def normalize_url(url: str) -> str:
    """Canonical form for comparing retrieval URLs.

    Ignores trivial variants (`www.` prefix, host casing, trailing slash,
    fragment) so a hit is never scored as a miss because the search engine
    and the golden set spell the same page differently. The scheme is kept:
    http/https mismatches are rare and worth noticing, not hiding.
    """
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parts.path.rstrip("/") or "/"
    return f"{scheme}://{host}{path}"


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant URLs present in the top-k retrieved URLs."""
    relevant_set = {normalize_url(url) for url in relevant}
    if not relevant_set:
        return 0.0
    top_k = [normalize_url(url) for url in retrieved][:k]
    hits = sum(1 for url in top_k if url in relevant_set)
    return hits / len(relevant_set)


def mrr(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Mean reciprocal rank (single query): 1 / rank of first relevant URL."""
    relevant_set = {normalize_url(url) for url in relevant}
    for rank, url in enumerate(retrieved, start=1):
        if normalize_url(url) in relevant_set:
            return 1.0 / rank
    return 0.0
