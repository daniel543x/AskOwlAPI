"""Alpha-weighted Reciprocal Rank Fusion (WU-4).

Fuses the vector-similarity leg and the lexical (FTS) leg of hybrid
retrieval: `score(d) = alpha * 1/(60 + rank_v(d)) + (1-alpha) * 1/(60 +
rank_l(d))`. `alpha = 1.0` degenerates to the pure vector ordering
(today's behavior — the lexical leg is not even queried), `0.0` to the
pure lexical ordering. The cross-encoder still reranks the fused pool;
RRF only decides which candidates make it into that pool and in which
order.
"""

RRF_K = 60


def _candidate_key(candidate: dict) -> str:
    """Identity across legs: chunk content (unique per content_hash)."""
    return str(candidate.get("content", ""))


def rrf_fuse(
    vector: list[dict],
    lexical: list[dict],
    alpha: float,
    k: int = RRF_K,
) -> list[dict]:
    """Fuse two ranked candidate lists into one (best first).

    Candidates present in both legs accumulate reciprocal-rank mass from
    each (weighted by alpha); single-leg candidates keep their leg's mass.
    The returned dicts are the original candidate objects (vector copies
    preferred on collision) — the downstream reranker is unchanged.
    """
    alpha = min(1.0, max(0.0, alpha))

    def rrf_map(candidates: list[dict]) -> dict[str, float]:
        return {
            _candidate_key(candidate): 1.0 / (k + rank)
            for rank, candidate in enumerate(candidates, start=1)
        }

    vector_scores = rrf_map(vector)
    lexical_scores = rrf_map(lexical)

    fused: list[tuple[float, str, dict]] = []
    seen: set[str] = set()
    for candidate in vector:
        key = _candidate_key(candidate)
        seen.add(key)
        score = (
            alpha * vector_scores[key]
            + (1.0 - alpha) * lexical_scores.get(key, 0.0)
        )
        fused.append((score, key, candidate))
    for candidate in lexical:
        key = _candidate_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        score = (
            alpha * vector_scores.get(key, 0.0)
            + (1.0 - alpha) * lexical_scores[key]
        )
        fused.append((score, key, candidate))

    fused.sort(key=lambda item: (-item[0], item[1]))
    # Zero fused score = the candidate came only from a leg whose weight
    # is 0 at this alpha. Dropping it keeps the extremes pure: alpha=1.0
    # yields exactly the vector pool, alpha=0.0 exactly the lexical pool.
    return [candidate for score, _, candidate in fused if score > 0.0]
