"""Pipeline pre-scores: decision-model opinions around retrieval.

Every helper degrades to "no opinion": a failing or unconfigured decision
backend must never break the ask pipeline (PLAN Phase 7). Toggles live in
settings and default to OFF until the eval harness blesses each use.
"""

import logging
from typing import Any

from ...modules.decision.base import DecisionError, IDecisionMaker
from ...tools.settings import settings

logger = logging.getLogger(__name__)

RELEVANCE_QUESTION = (
    "Does this search result address the user's question with "
    "on-topic content?"
)

CHUNK_CRITERIA = [
    "irrelevant",
    "weakly relevant",
    "relevant",
    "highly relevant",
    "essential",
]

QUERY_CHOICE_INSTRUCTIONS = (
    "Which query will find better web-search results for the user's "
    "question?"
)

GROUNDING_QUESTION = (
    "Is this answer fully supported by the provided context, with no "
    "claims beyond it?"
)


def _state(question: str, text: str) -> str:
    return f"Question: {question}\n\n{text}"


async def filter_search_results(
    maker: IDecisionMaker | None, query: str, results: list
) -> list:
    """Drop topically-irrelevant search results before scraping.

    Saves page fetches; keeps everything on any decision failure.
    """
    if maker is None or not results:
        return results

    try:
        states = [
            _state(query, f"{result.title}\n{result.snippet}\n{result.url}")
            for result in results
        ]
        decisions = await maker.batch_noul(RELEVANCE_QUESTION, states)
    except (DecisionError, Exception):  # noqa: BLE001 - degrade, never crash
        logger.warning(
            "Search-result pre-score failed; keeping all results.",
            exc_info=True,
        )
        return results

    threshold = settings.PRE_SCORE_RELEVANCE_THRESHOLD
    kept = [
        result
        for result, decision in zip(results, decisions, strict=True)
        if decision.yes >= threshold
    ]
    dropped = len(results) - len(kept)
    if dropped:
        logger.debug(
            "Search pre-score dropped %d/%d results as irrelevant.",
            dropped,
            len(results),
        )
    return kept if kept else results  # never return an empty list on opinion


async def filter_chunks(
    maker: IDecisionMaker | None, query: str, chunks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Second opinion next to the cross-encoder: 1-5 relevance scoring.

    Chunks scoring below PRE_SCORE_CHUNK_MIN_SCORE are dropped when the
    feature is enabled; failures keep every chunk.
    """
    if maker is None or not chunks:
        return chunks

    try:
        states = [
            _state(query, chunk.get("content", "")[:4000]) for chunk in chunks
        ]
        decisions = await maker.batch_score(
            "Rate how well this text answers the question.", CHUNK_CRITERIA, states
        )
    except (DecisionError, Exception):  # noqa: BLE001
        logger.warning("Chunk pre-score failed; keeping all chunks.", exc_info=True)
        return chunks

    minimum = settings.PRE_SCORE_CHUNK_MIN_SCORE
    kept = [
        chunk
        for chunk, decision in zip(chunks, decisions, strict=True)
        if decision.score >= minimum
    ]
    return kept if kept else chunks


async def choose_query(
    maker: IDecisionMaker | None, raw_query: str, optimized_query: str
) -> str:
    """Pick between the rewritten and the raw query; failures keep the
    optimizer's answer."""
    if maker is None or raw_query.strip() == optimized_query.strip():
        return optimized_query

    try:
        decision = await maker.choice(
            QUERY_CHOICE_INSTRUCTIONS,
            {"optimized": optimized_query, "raw": raw_query},
            _state(raw_query, f"optimized: {optimized_query}\nraw: {raw_query}"),
        )
    except (DecisionError, Exception):  # noqa: BLE001
        logger.warning("Query-choice pre-score failed; keeping optimized query.")
        return optimized_query

    if decision.choice in ("raw", "optimized"):
        logger.debug("Query choice pre-score picked '%s'.", decision.choice)
        return raw_query if decision.choice == "raw" else optimized_query
    return optimized_query


async def grounding_confidence(
    maker: IDecisionMaker | None, question: str, context: str, answer: str
) -> float | None:
    """P(answer is supported by the context); surfaced, never blocking."""
    if maker is None:
        return None
    try:
        decision = await maker.noul(
            GROUNDING_QUESTION,
            _state(question, f"Context:\n{context[:8000]}\n\nAnswer:\n{answer[:4000]}"),
        )
    except (DecisionError, Exception):  # noqa: BLE001
        logger.warning("Grounding pre-score failed; no grounding signal.")
        return None
    return decision.yes
