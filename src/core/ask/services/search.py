import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import HTTPException
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ValidationError

from ....modules.owl_reranker.base import IRanker
from ....modules.scrapy.providers.base import IScraper
from ....modules.search_engine.providers.base import ISearchProvider
from ....tools.settings import settings
from .chunking import chunk_markdown_data

logger = logging.getLogger(__name__)

# (event name, payload) pairs yielded by the pipeline and consumed by adapters.
PipelineEvent = tuple[str, dict[str, Any]]


# <----- Pydantic Schemas ----->
class SearchQuerySchema(BaseModel):
    search_query: str


query_parser = JsonOutputParser(pydantic_object=SearchQuerySchema)


# <----- Prompts ----->
query_optimization_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Web Search Query Optimizer API.
Your task is to convert the user's question into optimized search keywords.

Guidelines:
1. Noise Removal: Remove conversational filler.
2. Keyword Optimization: Focus on entities and nouns.
3. Language Preservation: Maintain original language.

{format_instructions}
""",
        ),
        ("human", "User Query: {query}"),
    ]
)

answer_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert AI assistant for a search engine.
Answer the user's question based strictly on the provided context chunks.

Guidelines:
1. Analyze the user's question and the provided context.
2. Formulate a precise answer based ONLY on the context.
3. You MUST cite sources using Markdown link format: [Source](URL) immediately
   after the claim.
4. If context doesn't contain the answer, state that clearly.

Context Chunks:
{data}""",
        ),
        ("human", "User Query: {query}"),
    ]
)


@asynccontextmanager
async def _stage(name: str):
    """Measure a pipeline stage; log the duration when debug mode is on."""
    start = time.perf_counter()
    try:
        yield
    finally:
        if settings.DEBUG:
            logger.debug(
                "Pipeline stage '%s' finished in %.3fs",
                name,
                time.perf_counter() - start,
            )


def _context_to_str(best_content: Any) -> str:
    if not best_content:
        return ""
    if isinstance(best_content, list) and isinstance(best_content[0], dict):
        return "\n\n---\n\n".join(
            item.get("content", str(item)) for item in best_content
        )
    return "\n\n---\n\n".join(str(item) for item in best_content)


# <----- Main Logic ----->
async def search_logic(
    query: str,
    searching: ISearchProvider,
    model: BaseChatModel,
    scraper: IScraper,
    ranker: IRanker,
) -> AsyncGenerator[PipelineEvent]:
    """Run the search pipeline, yielding (event, payload) pairs.

    Adapters (SSE / JSON) translate these events into transport-specific
    responses. Errors are reported as ``error`` events, never raised, so
    streaming consumers always get a well-formed termination.
    """
    try:
        # <----- LLM building query ----->
        yield "status", {"step": "Building query for search engine..."}

        actual_search_query = query
        async with _stage("query_optimization"):
            query_chain = query_optimization_prompt | model | query_parser
            try:
                parsed_output = await query_chain.ainvoke(
                    {
                        "query": query,
                        "format_instructions": query_parser.get_format_instructions(),
                    }
                )
                actual_search_query = parsed_output["search_query"]
            except (ValidationError, OutputParserException) as e:
                logger.warning(
                    "Query optimization failed, falling back to raw query: %s", e
                )
                actual_search_query = query

        yield "query_generated", {"query": actual_search_query}

        # <----- Searching ----->
        yield "status", {"step": "Searching URLs..."}
        async with _stage("web_search"):
            sources = await searching.search(query=actual_search_query)

        if not sources:
            yield "error", {"error": "No search results found."}
            return

        # <----- Ranking searching results ----->
        yield "status", {"step": "Ranking URLs..."}
        async with _stage("rank_sources"):
            ranked_sources = await asyncio.to_thread(
                ranker.rank_web_search, query, sources, settings.TOP_N_URLS
            )

        # <----- Scraping ----->
        yield "status", {"step": "Downloading data..."}
        async with _stage("scrape_pages"):
            content = await scraper.scrape(ranked_sources)

        if not content:
            yield (
                "warning",
                {
                    "warning": (
                        "No page content could be extracted. No answer can be given."
                    )
                },
            )
            return

        # <----- Chunking content data ----->
        yield "status", {"step": "Chunking data..."}
        async with _stage("chunk_pages"):
            chunked_content = await asyncio.to_thread(chunk_markdown_data, content)

        # <----- Ranking chunked data ----->
        yield "status", {"step": "Choose the best data..."}
        async with _stage("rank_chunks"):
            best_content = await asyncio.to_thread(
                ranker.rank_chunks, query, chunked_content, settings.TOP_N_CHUNKS
            )

        # <----- Context ready for answer generation ----->
        yield (
            "context_ready",
            {
                "context": _context_to_str(best_content),
                "sources": sources,
                "query": query,
            },
        )

    except Exception as e:
        logger.exception("Search pipeline failed for query: %s", query)
        yield "error", {"error": f"Critical error: {str(e)}"}


def _sse_frame(event: str, data: dict[str, Any]) -> str:
    """Format one Server-Sent Event frame with correct framing and terminator."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _astream_answer(
    query: str, context: str, model: BaseChatModel
) -> AsyncGenerator[str]:
    answer_chain = answer_generation_prompt | model
    async for chunk in answer_chain.astream({"query": query, "data": context}):
        if chunk.content:
            yield chunk.content


# <----- SSE Adapter ----->
async def search_sse_adapter(
    query: str,
    searching: ISearchProvider,
    model: BaseChatModel,
    scraper: IScraper,
    ranker: IRanker,
) -> AsyncGenerator[str]:
    """Translate pipeline events into a Server-Sent Events stream.

    Once streaming has started, errors MUST be emitted as ``error`` events;
    raising HTTPException here would kill the stream mid-flight.
    """
    async for event, data in search_logic(query, searching, model, scraper, ranker):
        if event == "context_ready":
            yield _sse_frame("status", {"step": "Generating answer..."})

            try:
                async for text in _astream_answer(query, data["context"], model):
                    yield _sse_frame("message", {"text": text})
            except Exception as e:
                logger.exception("Answer streaming failed for query: %s", query)
                yield _sse_frame(
                    "error", {"error": f"Answer generation failed: {str(e)}"}
                )
                return

            yield _sse_frame("done", {"status": "Completed"})
        else:
            yield _sse_frame(event, data)


# <----- JSON Adapter ----->
async def search_json_adapter(
    query: str,
    searching: ISearchProvider,
    model: BaseChatModel,
    scraper: IScraper,
    ranker: IRanker,
) -> dict[str, Any]:
    """Run the pipeline and return the full result as one JSON-ready dict."""
    search_data: dict[str, Any] | None = None
    error_detail: str | None = None

    async for event, data in search_logic(query, searching, model, scraper, ranker):
        if event == "error":
            error_detail = data.get("error", "Failed to execute search logic.")
            break
        if event == "context_ready":
            search_data = data
            break

    if error_detail:
        raise HTTPException(status_code=500, detail=error_detail)

    if not search_data:
        raise HTTPException(
            status_code=500,
            detail="No search data produced (no page content could be extracted).",
        )

    try:
        answer_chain = answer_generation_prompt | model
        response = await answer_chain.ainvoke(
            {"query": search_data["query"], "data": search_data["context"]}
        )
    except Exception as e:
        logger.exception("Answer generation failed for query: %s", query)
        raise HTTPException(
            status_code=500, detail=f"Answer generation failed: {str(e)}"
        ) from e

    return {
        "query": search_data["query"],
        "answer": response.content,
        "sources": search_data["sources"],
    }
