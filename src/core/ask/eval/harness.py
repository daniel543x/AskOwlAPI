"""Eval harness: measures retrieval quality of ask-pipeline configurations.

Baseline (Phase 2): cross-encoder ranking over the current request's chunks.
Vector (Phase 3): chunks embedded into the pgvector store at scrape time;
context retrieved as vector-similarity candidates, then cross-encoder
reranked. New retrieval configurations must pass here before becoming
defaults (PLAN Phase 3).

Runs against live services via `python -m src.core.ask.eval`; unit tests
exercise the same code path with fakes.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ....modules.decision.base import IDecisionMaker
from ....modules.owl_reranker.base import IRanker
from ....modules.scrapy.providers.base import IScraper
from ....modules.search_engine.providers.base import ISearchProvider, SearchResult
from ...tools.toolbox import AskToolbox, ChunkStoreLike, EmbedderLike
from .metrics import mrr, recall_at_k

logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"

# Tokens so common they say nothing about topical match.
_STOPWORDS = frozenset(
    """a an and are as at be but by for from how is it its of on or that the
    this to vs vs. what when where which who why with without""".split()
)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in "".join(
            c if c.isalnum() else " " for c in text.lower()
        ).split()
        if token not in _STOPWORDS and len(token) > 1
    }


def is_sane_result(query: str, title: str, snippet: str) -> bool:
    """True when a search result is topically compatible with the query.

    The Phase 3 eval was invalidated by search-egress poisoning (a Rust
    query returned peptide articles; k8s returned Gmail login pages). This
    heuristic keeps only results sharing at least one content token with
    the query (title or snippet), so nonsense results cannot enter the
    comparison even when an engine degrades to garbage.
    """
    query_tokens = _tokens(query)
    if not query_tokens:
        return True
    return bool(query_tokens & _tokens(f"{title} {snippet}"))


class SanityFilteringProvider(ISearchProvider):
    """Search wrapper dropping results topically unrelated to the query.

    Purely an eval-input guard: poisoned inputs make any A/B retrieval
    comparison meaningless. Production adoption may follow once measured.
    """

    def __init__(self, inner: ISearchProvider):
        self.inner = inner
        self.stats: dict[str, dict[str, int]] = {}

    async def search(
        self, query: str, engines: list[str] | None = None
    ) -> list[SearchResult]:
        results = await self.inner.search(query, engines=engines)
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


class CachedSearchProvider(ISearchProvider):
    """Memoizing search provider so every eval config sees identical results.

    Upstream engines flap between runs (CAPTCHAs, rate-limit suspensions);
    without a cache the baseline and vector configurations would be compared
    against different search results, which biases the verdict.
    """

    def __init__(self, inner: ISearchProvider):
        self.inner = inner
        self._cache: dict[str, list[SearchResult]] = {}
        self.query_count = 0

    async def search(
        self, query: str, engines: list[str] | None = None
    ) -> list[SearchResult]:
        if query not in self._cache:
            self._cache[query] = list(await self.inner.search(query))
            self.query_count += 1
        return list(self._cache[query])


class CachedScraper(IScraper):
    """Per-URL memoizing scraper: each page is fetched once per eval process.

    Only successful extractions (non-empty content) are cached, so transient
    fetch failures are retried on the next call instead of being frozen in.
    """

    def __init__(self, inner: IScraper):
        self.inner = inner
        self._cache: dict[str, dict[str, Any]] = {}
        self.fetch_count = 0

    async def scrape(
        self, data_from_search: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not data_from_search:
            return []
        misses = [
            item
            for item in data_from_search
            if item.get("url") not in self._cache
        ]
        self.fetch_count += len(misses)
        if misses:
            fetched = await self.inner.scrape(misses) or []
            for page in fetched:
                if page.get("url") and page.get("content"):
                    self._cache[page["url"]] = page
        return [
            dict(self._cache[item["url"]])
            if item.get("url") in self._cache
            else {"content": "", "url": item.get("url", "")}
            for item in data_from_search
        ]

    @property
    def name(self) -> str:
        return f"cached({self.inner.name})"


@dataclass
class EvalQuery:
    query: str
    relevant_urls: list[str]


@dataclass
class QueryResult:
    query: str
    context_urls: list[str]
    relevant_urls: list[str]
    recall: float
    reciprocal_rank: float
    latency_ms: float
    vector_retrieval: bool


@dataclass
class EvalReport:
    config_name: str
    results: list[QueryResult]
    mean_recall: float
    mean_mrr: float
    mean_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_golden_set(path: Path | None = None) -> list[EvalQuery]:
    source = path or GOLDEN_SET_PATH
    raw = json.loads(source.read_text())
    return [EvalQuery(**entry) for entry in raw["entries"]]


def _context_urls(selected_chunks: list[dict[str, Any]]) -> list[str]:
    """Ordered, de-duplicated source URLs of the selected context chunks."""
    urls: list[str] = []
    for chunk in selected_chunks:
        metadata = chunk.get("metadata") or {}
        url = metadata.get("url") or chunk.get("url") or ""
        if url and url not in urls:
            urls.append(url)
    return urls


async def evaluate_query(
    entry: EvalQuery,
    searching: ISearchProvider,
    scraper: IScraper,
    ranker: IRanker,
    embedder: EmbedderLike | None = None,
    chunk_repo: ChunkStoreLike | None = None,
    decision_maker: IDecisionMaker | None = None,
) -> QueryResult:
    """Run retrieval (no LLM) for one golden query and score it."""
    toolbox = AskToolbox(
        searching, scraper, ranker, embedder, chunk_repo,
        decision_maker=decision_maker,
    )

    start = time.perf_counter()
    await toolbox.web_search(entry.query)
    await toolbox.fetch_pages()
    await toolbox.get_context(entry.query)
    latency_ms = (time.perf_counter() - start) * 1000

    context_urls = _context_urls(toolbox.selected_chunks)
    return QueryResult(
        query=entry.query,
        context_urls=context_urls,
        relevant_urls=entry.relevant_urls,
        recall=recall_at_k(context_urls, entry.relevant_urls, k=len(context_urls)),
        reciprocal_rank=mrr(context_urls, entry.relevant_urls),
        latency_ms=latency_ms,
        vector_retrieval=toolbox.used_vector_retrieval,
    )


async def run_eval(
    config_name: str,
    entries: list[EvalQuery],
    searching: ISearchProvider,
    scraper: IScraper,
    ranker: IRanker,
    embedder: EmbedderLike | None = None,
    chunk_repo: ChunkStoreLike | None = None,
    decision_maker: IDecisionMaker | None = None,
) -> EvalReport:
    """Evaluate one configuration over the whole golden set."""
    results = [
        await evaluate_query(
            entry, searching, scraper, ranker, embedder, chunk_repo, decision_maker
        )
        for entry in entries
    ]
    count = len(results) or 1
    return EvalReport(
        config_name=config_name,
        results=results,
        mean_recall=sum(r.recall for r in results) / count,
        mean_mrr=sum(r.reciprocal_rank for r in results) / count,
        mean_latency_ms=sum(r.latency_ms for r in results) / count,
    )


def comparison_verdict(
    baseline: EvalReport, vector: EvalReport
) -> tuple[bool, str]:
    """Gate for the Phase 3 DoD ("vector must not lose to the baseline").

    A vector run in which any query silently fell back to in-memory ranking
    is invalid: it compares the baseline against itself and proves nothing.
    """
    degraded = sum(1 for r in vector.results if not r.vector_retrieval)
    if degraded:
        return False, (
            f"vector config fell back to baseline ranking on {degraded}/"
            f"{len(vector.results)} queries (embedder or chunk store "
            "failure); fix the underlying error and re-run"
        )
    better = vector.mean_recall >= baseline.mean_recall
    return better, f"recall {vector.mean_recall:.3f} vs {baseline.mean_recall:.3f}"


def format_report(report: EvalReport) -> str:
    lines = [
        f"Eval report: {report.config_name}",
        f"  queries:          {len(report.results)}",
        f"  mean recall@n:    {report.mean_recall:.3f}",
        f"  mean MRR:         {report.mean_mrr:.3f}",
        f"  mean latency ms:  {report.mean_latency_ms:.0f}",
    ]
    for result in report.results:
        lines.append(
            f"    recall={result.recall:.2f} mrr={result.reciprocal_rank:.2f} "
            f"vector={result.vector_retrieval} {result.query[:60]}"
        )
    return "\n".join(lines)
