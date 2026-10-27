"""Pipeline internals promoted to first-class tools.

The pipeline calls these directly; Phase 5 binds the same operations to
agentic models via `build_tools`. Tools share one `AskToolbox` instance per
request so results (sources, pages, chunks, context) flow between steps.

When an embedder and chunk store are available, `fetch_pages` embeds the
chunks at scrape time and `get_context` retrieves vector-similarity
candidates before cross-encoder reranking (hybrid retrieval). Both degrade
gracefully to in-memory ranking when embedding is unavailable.
"""

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Protocol
from urllib.parse import urlparse

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ...modules.chunk_store.model import chunk_hash
from ...modules.decision.base import IDecisionMaker
from ...modules.owl_reranker.base import IRanker
from ...modules.scrapy.providers.base import IScraper
from ...modules.search_engine.fusion import apply_fusion_floor, merge_results
from ...modules.search_engine.providers.base import ISearchProvider
from ...modules.search_engine.sanity import is_sane_result
from ...tools.settings import settings
from .chunking import chunk_markdown_data
from .pre_scores import filter_chunks, filter_search_results
from .recency import apply_recency_bias

logger = logging.getLogger(__name__)


class EmbedderLike(Protocol):
    """Embedder surface the toolbox needs; IEmbedder satisfies it."""

    @property
    def dims(self) -> int: ...

    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class ChunkStoreLike(Protocol):
    """Chunk-store surface the toolbox needs; ChunkRepository satisfies it."""

    async def store_chunks(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
        conversation_id: uuid.UUID | None = None,
    ) -> int: ...

    async def existing_hashes(self, hashes: list[str]) -> set[str]: ...

    async def vector_candidates(
        self,
        query_embedding: list[float],
        k: int,
        conversation_id: uuid.UUID | None = None,
    ) -> list[dict[str, Any]]: ...


class WebSearchArgs(BaseModel):
    query: str = Field(description="Search query to run against the web.")


class FetchPagesArgs(BaseModel):
    urls: list[str] | None = Field(
        default=None,
        description=(
            "URLs to fetch. Omit to fetch the top-ranked search results."
        ),
    )


class GetContextArgs(BaseModel):
    query: str = Field(
        description="Query used to rank chunks and select the best context."
    )


class AskToolbox:
    """Web search / fetch / context-selection tools with shared state."""

    def __init__(
        self,
        searching: ISearchProvider,
        scraper: IScraper,
        ranker: IRanker,
        embedder: EmbedderLike | None = None,
        chunk_repo: ChunkStoreLike | None = None,
        allowed_domains: list[str] | None = None,
        allowed_engines: list[str] | None = None,
        decision_maker: IDecisionMaker | None = None,
        max_urls: int | None = None,
        conversation_id: uuid.UUID | None = None,
        scope: str = "global",
        recency_bias: float = 0.0,
        now: datetime | None = None,
    ):
        self.searching = searching
        self.scraper = scraper
        self.ranker = ranker
        self.embedder = embedder
        self.chunk_repo = chunk_repo
        self.decision_maker = decision_maker
        # Persona constraints: None/empty = unconstrained.
        self.allowed_domains = [d.lower() for d in (allowed_domains or [])]
        self.allowed_engines = allowed_engines or None
        # Per-request URL-selection cap (WU-6); None = settings default,
        # always clamped to the operator ceiling TOP_N_URLS_MAX.
        self.max_urls = max(
            1, min(max_urls or settings.TOP_N_URLS, settings.TOP_N_URLS_MAX)
        )
        # Retrieval scope (WU-5): "global" = today's behavior; "conversation"
        # filters vector candidates to this conversation (+ legacy NULL
        # rows); "request" restricts retrieval to this turn's fetched pages.
        self.conversation_id = conversation_id
        self.scope = scope
        # Recency boost (WU-11): clamped to the operator ceiling; `now` is
        # injectable so tests can freeze the clock.
        self.recency_bias = max(
            0.0, min(recency_bias, settings.RECENCY_BIAS_MAX)
        )
        self.now = now or datetime.now(UTC)

        self.raw_results: list = []
        self.ranked_sources: list[dict[str, Any]] = []
        self.pages: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []
        self.selected_chunks: list[dict[str, Any]] = []
        self.citations: list[dict[str, Any]] = []
        self.used_vector_retrieval = False
        # Hygiene stats surfaced in the debug SSE event (WU-3).
        self.search_stats: dict[str, int] = {}

    async def web_search(self, query: str) -> str:
        """Search the web and rank the results; returns JSON for the LLM.

        Hygiene pipeline (AR-2): sanity-drop topically unrelated results,
        merge engine duplicates by normalized URL, fusion-score the merged
        pool, apply the fusion floor (never keeping fewer than three), then
        cross-encoder rank to the URL cap with fusion as the tie-breaker.
        """
        self.raw_results = await self.searching.search(
            query, engines=self.allowed_engines
        )
        return await self._rank_results(query)

    async def web_search_many(self, queries: list[str]) -> str:
        """Fan out query variants (WU-2) and rank the merged pool.

        The variants run against the provider unchanged; the AR-2 merge
        collapses the duplicates so ranking cost does not multiply. The
        first query is the primary one (sanity filter + ranking use it).
        """
        results: list = []
        for query in queries:
            results.extend(
                await self.searching.search(query, engines=self.allowed_engines)
            )
        self.raw_results = results
        return await self._rank_results(queries[0])

    async def _rank_results(self, query: str) -> str:
        """Sanity, dedup+fusion, floor, and ranking over `raw_results`."""
        if self.allowed_domains:
            self.raw_results = self._filter_domains(self.raw_results)

        dropped_sanity = 0
        if settings.RESULT_SANITY_FILTER:
            kept = [
                result
                for result in self.raw_results
                if is_sane_result(query, result.title, result.snippet)
            ]
            dropped_sanity = len(self.raw_results) - len(kept)
            self.raw_results = kept

        if settings.PRE_SCORE_SEARCH_FILTER and self.decision_maker:
            self.raw_results = await filter_search_results(
                self.decision_maker, query, self.raw_results
            )

        merged = merge_results(self.raw_results, settings.DOMAIN_AUTHORITY)
        merged, dropped_floor = apply_fusion_floor(
            merged, settings.URL_SELECTION_MIN_FUSION
        )

        ranked = await asyncio.to_thread(
            self.ranker.rank_web_search,
            query,
            [item.as_search_result() for item in merged],
            self.max_urls,
        )

        self.search_stats = {
            "dropped_results": dropped_sanity + dropped_floor,
            "merged_results": len(self.raw_results) - len(merged),
            "max_urls": self.max_urls,
        }

        by_url = {item.url: item for item in merged}

        def _fusion(url: str) -> float:
            entry = by_url.get(url)
            return entry.fusion_score if entry else 0.0

        if settings.FUSION_TIEBREAK:
            # Cross-encoder stays primary; ties at the third decimal (MiniLM
            # saturates around 0.999x) are broken by the fusion score.
            # reverse=True keeps ties stable (original order preserved).
            ranked = sorted(
                ranked,
                key=lambda item: (
                    round(float(item.get("score", 0.0) or 0.0), 3),
                    _fusion(item["url"]),
                ),
                reverse=True,
            )

        self.ranked_sources = [
            {
                "url": item["url"],
                "title": by_url[item["url"]].title
                if item["url"] in by_url
                else "",
                "score": item["score"],
                "fusion_score": round(by_url[item["url"]].fusion_score, 4)
                if item["url"] in by_url
                else None,
            }
            for item in ranked
        ]
        return json.dumps(self.ranked_sources)

    def _filter_domains(self, results: list) -> list:
        """Keep only results whose host matches an allowed domain (suffix)."""
        kept = []
        for result in results:
            host = (urlparse(result.url).hostname or "").lower()
            if any(
                host == domain or host.endswith(f".{domain}")
                for domain in self.allowed_domains
            ):
                kept.append(result)
        dropped = len(results) - len(kept)
        if dropped:
            logger.debug(
                "Domain constraint dropped %d of %d search results.",
                dropped,
                len(results),
            )
        return kept

    async def fetch_pages(self, urls: list[str] | None = None) -> str:
        """Download pages and split them into chunks; returns page previews.

        With an embedder configured, chunks are embedded and persisted to the
        chunk store (duplicates skipped). Failures degrade to the in-memory
        path; they never break the pipeline.
        """
        targets = urls or [item["url"] for item in self.ranked_sources]
        if not targets:
            return "No URLs to fetch."

        self.pages = await self.scraper.scrape(
            [{"url": url} for url in targets]
        ) or []
        self.chunks = await asyncio.to_thread(
            chunk_markdown_data,
            [page for page in self.pages if page.get("content")],
        )

        if self.embedder and self.chunk_repo and self.chunks:
            try:
                await self._embed_and_store()
            except Exception:
                logger.warning(
                    "Embedding chunks failed; continuing without the vector "
                    "store for this request.",
                    exc_info=True,
                )

        previews = [
            f"# {page['url']}\n{page['content'][:500]}"
            for page in self.pages
            if page.get("content")
        ]
        return "\n\n---\n\n".join(previews) if previews else "No content extracted."

    async def _embed_and_store(self) -> None:
        """Embed and persist chunks, skipping already-stored ones (WU-10).

        The hash is computed exactly like `ChunkRepository.store_chunks`, so
        a re-visit of the same page re-embeds nothing.
        """
        assert self.embedder and self.chunk_repo

        def _hashes(chunks: list[dict[str, Any]]) -> list[str]:
            return [
                chunk_hash(
                    (chunk.get("metadata") or {}).get("url", "")
                    or chunk.get("url", ""),
                    chunk["content"],
                )
                for chunk in chunks
            ]

        try:
            existing = await self.chunk_repo.existing_hashes(
                _hashes(self.chunks)
            )
        except Exception:
            # Skip-check failure must not skip the store entirely; fall
            # through and let store_chunks' own conflict handling dedupe.
            logger.warning(
                "Embed-skip hash lookup failed; embedding all chunks.",
                exc_info=True,
            )
            existing = set()

        todo = [
            chunk
            for chunk, digest in zip(
                self.chunks, _hashes(self.chunks), strict=True
            )
            if digest not in existing
        ]
        if not todo:
            logger.debug("Embed-skip: all %d chunks already stored.", len(self.chunks))
            return

        embeddings = await self.embedder.embed_texts(
            [chunk["content"] for chunk in todo]
        )
        inserted = await self.chunk_repo.store_chunks(
            todo,
            embeddings,
            conversation_id=self.conversation_id,
        )
        logger.debug(
            "Stored %d new chunks in the vector store (%d skipped).",
            inserted,
            len(self.chunks) - len(todo),
        )

    async def get_context(self, query: str) -> str:
        """Select the context chunks for the query.

        Hybrid retrieval: vector-similarity candidates from the chunk store
        (top-k) refined by the cross-encoder (top-n). Falls back to ranking
        the current request's chunks when embedding is unavailable or
        retrieval fails.
        """
        candidates: list[dict[str, Any]] | None = None

        if self.scope != "request" and self.embedder and self.chunk_repo:
            try:
                candidates = await self._vector_candidates(query)
            except Exception:
                logger.warning(
                    "Vector retrieval failed; falling back to in-memory "
                    "chunk ranking.",
                    exc_info=True,
                )

        if candidates is not None:
            self.used_vector_retrieval = True
            pool = candidates
        else:
            pool = self.chunks

        self.selected_chunks = await asyncio.to_thread(
            self.ranker.rank_chunks, query, pool, settings.TOP_N_CHUNKS
        )
        if settings.PRE_SCORE_CHUNK_FILTER and self.decision_maker:
            self.selected_chunks = await filter_chunks(
                self.decision_maker, query, self.selected_chunks
            )
        if self.recency_bias > 0:
            self.selected_chunks = apply_recency_bias(
                self.selected_chunks, self.recency_bias, self.now
            )
        return self._numbered_context()

    def _numbered_context(self) -> str:
        """Join chunks as `[n] content` and record the citation list (WU-9).

        The answer contract cites these numbers; `validate_citations`
        resolves them against `self.citations` after generation.
        """
        parts: list[str] = []
        citations: list[dict[str, Any]] = []
        for index, chunk in enumerate(self.selected_chunks, start=1):
            parts.append(f"[{index}] {chunk.get('content', '')}")
            metadata = chunk.get("metadata") or {}
            citations.append(
                {
                    "n": index,
                    "url": metadata.get("url") or chunk.get("url") or "",
                    "title": metadata.get("title") or "",
                }
            )
        self.citations = citations
        return "\n\n---\n\n".join(parts)

    async def _vector_candidates(self, query: str) -> list[dict[str, Any]]:
        assert self.embedder and self.chunk_repo
        query_embedding = (await self.embedder.embed_texts([query]))[0]
        return await self.chunk_repo.vector_candidates(
            query_embedding,
            settings.VECTOR_CANDIDATES_K,
            conversation_id=(
                self.conversation_id if self.scope == "conversation" else None
            ),
        )


class PreScoreArgs(BaseModel):
    question: str = Field(description="Yes/no question about the text.")
    text: str = Field(description="Text the question refers to.")


async def _pre_score(toolbox: AskToolbox, question: str, text: str) -> str:
    """Typed yes/no decision with a calibrated probability (JSON)."""
    if toolbox.decision_maker is None:
        return json.dumps(
            {"error": "No decision model configured; decide without it."}
        )
    try:
        decision = await toolbox.decision_maker.noul(question, text)
    except Exception:  # noqa: BLE001 - degrade to no-opinion
        return json.dumps({"error": "Decision backend failed."})
    return json.dumps(
        {"yes": round(decision.yes, 4), "confidence": round(decision.confidence, 4)}
    )


def build_tools(toolbox: AskToolbox) -> list[StructuredTool]:
    """Bindable tool schemas for agentic use (Phase 5+).

    `pre_score` appears only when a decision model is configured.
    """
    tools: list[StructuredTool] = []
    if toolbox.decision_maker is not None:
        tools.append(
            StructuredTool.from_function(
                coroutine=lambda **kwargs: _pre_score(
                    toolbox, kwargs["question"], kwargs["text"]
                ),
                name="pre_score",
                description=(
                    "Ask a calibrated decision model a yes/no question "
                    "about a text; returns P(yes) with confidence. Use for "
                    "cheap relevance/grounding checks before fetching or "
                    "citing."
                ),
                args_schema=PreScoreArgs,
            )
        )
    tools.extend(
        [
            StructuredTool.from_function(
                coroutine=toolbox.web_search,
                name="web_search",
                description="Search the web and return ranked results.",
                args_schema=WebSearchArgs,
            ),
            StructuredTool.from_function(
                coroutine=toolbox.fetch_pages,
                name="fetch_pages",
                description="Fetch pages (default: top-ranked results) as markdown.",
                args_schema=FetchPagesArgs,
            ),
            StructuredTool.from_function(
                coroutine=toolbox.get_context,
                name="get_context",
                description="Select the chunks most relevant to the query as context.",
                args_schema=GetContextArgs,
            ),
        ]
    )
    return tools
