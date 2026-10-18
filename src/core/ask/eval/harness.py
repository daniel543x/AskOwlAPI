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

from ....modules.chunk_store.repository import ChunkRepository
from ....modules.embedding.base import IEmbedder
from ....modules.owl_reranker.base import IRanker
from ....modules.scrapy.providers.base import IScraper
from ....modules.search_engine.providers.base import ISearchProvider
from ...tools.toolbox import AskToolbox
from .metrics import mrr, recall_at_k

logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"


class CachedSearchProvider:
    """Memoizing search provider so every eval config sees identical results.

    Upstream engines flap between runs (CAPTCHAs, rate-limit suspensions);
    without a cache the baseline and vector configurations would be compared
    against different search results, which biases the verdict.
    """

    def __init__(self, inner: ISearchProvider):
        self.inner = inner
        self._cache: dict[str, list] = {}
        self.query_count = 0

    async def search(self, query: str, engines: list[str] | None = None) -> list:
        if query not in self._cache:
            self._cache[query] = list(await self.inner.search(query=query))
            self.query_count += 1
        return list(self._cache[query])


class CachedScraper:
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
    embedder: IEmbedder | None = None,
    chunk_repo: ChunkRepository | None = None,
) -> QueryResult:
    """Run retrieval (no LLM) for one golden query and score it."""
    toolbox = AskToolbox(searching, scraper, ranker, embedder, chunk_repo)

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
    embedder: IEmbedder | None = None,
    chunk_repo: ChunkRepository | None = None,
) -> EvalReport:
    """Evaluate one configuration over the whole golden set."""
    results = [
        await evaluate_query(entry, searching, scraper, ranker, embedder, chunk_repo)
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
