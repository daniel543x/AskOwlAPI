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
from typing import Any
from urllib.parse import urlparse

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ...modules.chunk_store.repository import ChunkRepository
from ...modules.embedding.base import IEmbedder
from ...modules.owl_reranker.base import IRanker
from ...modules.scrapy.providers.base import IScraper
from ...modules.search_engine.providers.base import ISearchProvider
from ...tools.settings import settings
from .chunking import chunk_markdown_data

logger = logging.getLogger(__name__)


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
        embedder: IEmbedder | None = None,
        chunk_repo: ChunkRepository | None = None,
        allowed_domains: list[str] | None = None,
        allowed_engines: list[str] | None = None,
    ):
        self.searching = searching
        self.scraper = scraper
        self.ranker = ranker
        self.embedder = embedder
        self.chunk_repo = chunk_repo
        # Persona constraints: None/empty = unconstrained.
        self.allowed_domains = [d.lower() for d in (allowed_domains or [])]
        self.allowed_engines = allowed_engines or None

        self.raw_results: list = []
        self.ranked_sources: list[dict[str, Any]] = []
        self.pages: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []
        self.selected_chunks: list[dict[str, Any]] = []
        self.used_vector_retrieval = False

    async def web_search(self, query: str) -> str:
        """Search the web and rank the results; returns JSON for the LLM."""
        self.raw_results = await self.searching.search(
            query, engines=self.allowed_engines
        )

        if self.allowed_domains:
            self.raw_results = self._filter_domains(self.raw_results)

        ranked = await asyncio.to_thread(
            self.ranker.rank_web_search, query, self.raw_results, settings.TOP_N_URLS
        )

        titles = {result.url: result.title for result in self.raw_results}
        self.ranked_sources = [
            {
                "url": item["url"],
                "title": titles.get(item["url"], ""),
                "score": item["score"],
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

        self.pages = await self.scraper.scrape([{"url": url} for url in targets])
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
        assert self.embedder and self.chunk_repo
        embeddings = await self.embedder.embed_texts(
            [chunk["content"] for chunk in self.chunks]
        )
        inserted = await self.chunk_repo.store_chunks(self.chunks, embeddings)
        logger.debug("Stored %d new chunks in the vector store.", inserted)

    async def get_context(self, query: str) -> str:
        """Select the context chunks for the query.

        Hybrid retrieval: vector-similarity candidates from the chunk store
        (top-k) refined by the cross-encoder (top-n). Falls back to ranking
        the current request's chunks when embedding is unavailable or
        retrieval fails.
        """
        candidates: list[dict[str, Any]] | None = None

        if self.embedder and self.chunk_repo:
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
        return "\n\n---\n\n".join(
            chunk.get("content", "") for chunk in self.selected_chunks
        )

    async def _vector_candidates(self, query: str) -> list[dict[str, Any]]:
        assert self.embedder and self.chunk_repo
        query_embedding = (await self.embedder.embed_texts([query]))[0]
        return await self.chunk_repo.vector_candidates(
            query_embedding, settings.VECTOR_CANDIDATES_K
        )


def build_tools(toolbox: AskToolbox) -> list[StructuredTool]:
    """Bindable tool schemas for agentic use (Phase 5)."""
    return [
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
