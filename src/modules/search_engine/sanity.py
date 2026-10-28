"""Search-result sanity filtering (AR-2 / WU-3).

Drops results topically unrelated to the query before they can cost a page
fetch or pollute the context. Originally an eval-input guard against the
Phase 3 search-egress poisoning (a Rust query returned peptide articles);
validated by the Phase-3 eval re-run and promoted to the production default.
"""

import logging

from .providers.base import ISearchProvider, SearchResult

logger = logging.getLogger(__name__)

# Tokens so common they say nothing about topical match.
_STOPWORDS = frozenset(
    """a an and are as at be but by for from how is it its of on or that the
    this to vs vs. what when where which who why with without""".split()
)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in "".join(
            char if char.isalnum() else " " for char in text.lower()
        ).split()
        if token not in _STOPWORDS and len(token) > 1
    }


def is_sane_result(query: str, title: str, snippet: str) -> bool:
    """True when a search result is topically compatible with the query.

    Keeps only results sharing at least one content token with the query
    (title or snippet). A query with no content tokens keeps everything.
    """
    query_tokens = _tokens(query)
    if not query_tokens:
        return True
    return bool(query_tokens & _tokens(f"{title} {snippet}"))


class SanityFilteringProvider(ISearchProvider):
    """Search wrapper dropping results topically unrelated to the query.

    Used by the eval harness as an input guard; the production pipeline
    applies the same `is_sane_result` check inside `AskToolbox.web_search`
    (`RESULT_SANITY_FILTER`). This class remains for standalone wrapping and
    per-query drop statistics.
    """

    def __init__(self, inner: ISearchProvider):
        self.inner = inner
        self.stats: dict[str, dict[str, int]] = {}

    async def search(
        self,
        query: str,
        engines: list[str] | None = None,
        categories: str | None = None,
    ) -> list[SearchResult]:
        results = await self.inner.search(
            query, engines=engines, categories=categories
        )
        kept = [
            result
            for result in results
            if is_sane_result(query, result.title, result.snippet)
        ]
        entry = self.stats.setdefault(
            query, {"returned": 0, "kept": 0, "dropped": 0}
        )
        entry["returned"] += len(results)
        entry["kept"] += len(kept)
        entry["dropped"] += len(results) - len(kept)
        return kept

    def poisoned_queries(self) -> list[str]:
        """Queries where filtering removed everything (egress poisoning)."""
        return [
            query
            for query, entry in self.stats.items()
            if entry["returned"] > 0 and entry["kept"] == 0
        ]
