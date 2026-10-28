"""Recency boosting for chunk ranking (WU-11 / AR-7).

Opt-in multiplicative boost on the cross-encoder chunk score by publication
age: `final = base * (1 + bias * recency)` with recency 1.0 (<30 days),
0.6 (<1 year), 0.3 (<3 years), 0.0 (older or unknown). Unknown dates are
never penalized — the boost only adds. Applied after `rank_chunks`, so it
re-orders the selected chunks; the cross-encoder stays the base signal.
"""

import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

_DAY = 86400.0
# (max age in days, recency weight), most recent first.
_RECENCY_TIERS: tuple[tuple[float, float], ...] = (
    (30.0, 1.0),
    (365.0, 0.6),
    (365.0 * 3, 0.3),
)


def _parse(published_at: str | None) -> datetime | None:
    if not published_at:
        return None
    try:
        parsed = datetime.fromisoformat(str(published_at)[:10])
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def recency_weight(published_at: str | None, now: datetime) -> float:
    """0..1 recency weight of a publication date (0 for unknown)."""
    published = _parse(published_at)
    if published is None:
        return 0.0
    age_days = max(0.0, (now - published).total_seconds() / _DAY)
    for max_age, weight in _RECENCY_TIERS:
        if age_days < max_age:
            return weight
    return 0.0


def apply_recency_bias(
    chunks: list[dict], bias: float, now: datetime
) -> list[dict]:
    """Re-score and re-order selected chunks by publication recency.

    The cross-encoder score lives in `metadata.relevance_score`; chunks
    without one keep their relative order (stable sort on equal keys).
    """
    if bias <= 0:
        return chunks

    scored: list[tuple[float, int, dict]] = []
    for index, chunk in enumerate(chunks):
        metadata = chunk.get("metadata") or {}
        base = float(metadata.get("relevance_score") or 0.0)
        weight = recency_weight(metadata.get("published_at"), now)
        adjusted = base * (1.0 + bias * weight)
        metadata["relevance_score"] = adjusted
        chunk["metadata"] = metadata
        scored.append((adjusted, index, chunk))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [chunk for _, _, chunk in scored]
