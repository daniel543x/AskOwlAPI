"""Search-result dedup and engine-agreement fusion scoring (AR-2 / WU-3).

Deterministic, model-free: results from multiple engines (or multiple query
variants) are grouped by normalized URL and scored by engine agreement,
domain authority, and original position. The cross-encoder stays the primary
ranking signal; the fusion score is its tie-breaker (its absolute value only
becomes a meaningful floor once calibrated on the fused pool).
"""

import logging
from dataclasses import dataclass
from urllib.parse import urlparse

from ...tools.urls import normalize_url
from .providers.base import SearchResult

logger = logging.getLogger(__name__)

AGREEMENT_WEIGHT = 0.4
AUTHORITY_WEIGHT = 0.3
POSITION_WEIGHT = 0.3
# Agreement saturates once this many distinct engines agreed on a result.
AGREEMENT_NORMALIZER = 3.0
# Results below this many are always kept (degrade-don't-empty rule).
MIN_KEPT_RESULTS = 3


@dataclass
class MergedResult:
    """One result after URL-normalized dedup, with fusion scoring inputs."""

    url: str  # normalized URL
    title: str
    snippet: str
    merged_count: int
    original_rank: int  # best (lowest) 0-based position before the merge
    fusion_score: float

    def as_search_result(self) -> SearchResult:
        return SearchResult(title=self.title, snippet=self.snippet, url=self.url)


def authority_for(host: str, authority: dict[str, float]) -> float:
    """Domain-authority score for a host; 0.5 for unknown hosts.

    Map keys may be registrable domains (`wikipedia.org`, matching the host
    or any subdomain) or dot-leading TLD suffixes (`.gov`, `.edu`).
    """
    host = host.lower()
    for key, value in authority.items():
        key = key.lower()
        if key.startswith("."):
            if host.endswith(key):
                return value
        elif host == key or host.endswith(f".{key}"):
            return value
    return 0.5


def _fusion(
    merged_count: int, original_rank: int, host: str, authority: dict[str, float]
) -> float:
    agreement = min(merged_count / AGREEMENT_NORMALIZER, 1.0)
    position = 1.0 / (1.0 + original_rank / 10.0)
    return (
        AGREEMENT_WEIGHT * agreement
        + AUTHORITY_WEIGHT * authority_for(host, authority)
        + POSITION_WEIGHT * position
    )


def merge_results(
    results: list[SearchResult], authority: dict[str, float]
) -> list[MergedResult]:
    """Group results by normalized URL and score the fused entries.

    Keeps first-appearance order. For each group the longest title and
    snippet survive (more signal for the sanity filter and the LLM), the
    merged count records engine agreement, and the best original rank feeds
    the position term.
    """
    groups: dict[str, dict] = {}
    order: list[str] = []
    for rank, result in enumerate(results):
        url = normalize_url(result.url)
        if url not in groups:
            groups[url] = {
                "title": result.title,
                "snippet": result.snippet,
                "merged_count": 0,
                "original_rank": rank,
            }
            order.append(url)
        group = groups[url]
        group["merged_count"] += 1
        if len(result.title) > len(group["title"]):
            group["title"] = result.title
        if len(result.snippet) > len(group["snippet"]):
            group["snippet"] = result.snippet

    merged: list[MergedResult] = []
    for url in order:
        group = groups[url]
        host = urlparse(url).hostname or ""
        merged.append(
            MergedResult(
                url=url,
                title=group["title"],
                snippet=group["snippet"],
                merged_count=group["merged_count"],
                original_rank=group["original_rank"],
                fusion_score=_fusion(
                    group["merged_count"], group["original_rank"], host, authority
                ),
            )
        )
    return merged


def apply_fusion_floor(
    merged: list[MergedResult], floor: float
) -> tuple[list[MergedResult], int]:
    """Drop results scoring below `floor`; never keep fewer than three.

    Returns (kept, dropped_count). The floor is meant for calibrated fusion
    scores; the keep-at-least-three rule preserves the degrade-don't-empty
    guarantee from `pre_scores.py` (rescued entries are the best-scoring
    drops, in their original order).
    """
    if floor <= 0.0:
        return merged, 0
    keep_idx = [i for i, m in enumerate(merged) if m.fusion_score >= floor]
    minimum = min(MIN_KEPT_RESULTS, len(merged))
    if len(keep_idx) < minimum:
        dropped_idx = sorted(
            (i for i in range(len(merged)) if i not in keep_idx),
            key=lambda i: merged[i].fusion_score,
            reverse=True,
        )
        keep_idx.extend(dropped_idx[: minimum - len(keep_idx)])
        keep_idx.sort()
    kept = [merged[i] for i in keep_idx]
    return kept, len(merged) - len(kept)
