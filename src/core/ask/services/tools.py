"""Pipeline internals promoted to first-class tools.

The pipeline calls these directly; Phase 5 binds the same operations to
agentic models via `build_tools`. Tools share one `AskToolbox` instance per
request so results (sources, pages, chunks, context) flow between steps.
"""

import asyncio
import json
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ....modules.owl_reranker.base import IRanker
from ....modules.scrapy.providers.base import IScraper
from ....modules.search_engine.providers.base import ISearchProvider
from ....tools.settings import settings
from .chunking import chunk_markdown_data


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

    def __init__(self, searching: ISearchProvider, scraper: IScraper, ranker: IRanker):
        self.searching = searching
        self.scraper = scraper
        self.ranker = ranker

        self.raw_results: list = []
        self.ranked_sources: list[dict[str, Any]] = []
        self.pages: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []
        self.selected_chunks: list[dict[str, Any]] = []

    async def web_search(self, query: str) -> str:
        """Search the web and rank the results; returns JSON for the LLM."""
        self.raw_results = await self.searching.search(query=query)

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

    async def fetch_pages(self, urls: list[str] | None = None) -> str:
        """Download pages and split them into chunks; returns page previews."""
        targets = urls or [item["url"] for item in self.ranked_sources]
        if not targets:
            return "No URLs to fetch."

        self.pages = await self.scraper.scrape([{"url": url} for url in targets])
        self.chunks = await asyncio.to_thread(
            chunk_markdown_data,
            [page for page in self.pages if page.get("content")],
        )

        previews = [
            f"# {page['url']}\n{page['content'][:500]}"
            for page in self.pages
            if page.get("content")
        ]
        return "\n\n---\n\n".join(previews) if previews else "No content extracted."

    async def get_context(self, query: str) -> str:
        """Rerank chunks for the query and return the selected context."""
        self.selected_chunks = await asyncio.to_thread(
            self.ranker.rank_chunks, query, self.chunks, settings.TOP_N_CHUNKS
        )
        return "\n\n---\n\n".join(
            chunk.get("content", "") for chunk in self.selected_chunks
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
