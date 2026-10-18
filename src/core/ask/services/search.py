"""Ask pipeline (protocol v2): every LLM step goes through a tool call.

`search_logic` runs the linear pipeline (rewrite -> web_search ->
fetch_pages -> get_context) and yields (event, payload) pairs. Adapters
translate the events into SSE (with event ids) or one JSON document.
"""

import json
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import HTTPException
from langchain_core.language_models import BaseChatModel

from ....modules.chunk_store.repository import ChunkRepository
from ....modules.embedding.base import IEmbedder
from ....modules.owl_reranker.base import IRanker
from ....modules.scrapy.providers.base import IScraper
from ....modules.search_engine.providers.base import ISearchProvider
from ....tools.settings import settings
from ...tools.toolbox import AskToolbox
from .llm_steps import (
    ToolProtocolError,
    generate_cited_answer,
    rewrite_query,
    stream_cited_answer,
)

logger = logging.getLogger(__name__)

# (event name, payload) pairs yielded by the pipeline and consumed by adapters.
PipelineEvent = tuple[str, dict[str, Any]]

NO_CONTEXT_ANSWER = (
    "No relevant content could be extracted from the search results, "
    "so no answer can be given."
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


# <----- Main Logic ----->
async def search_logic(
    query: str,
    searching: ISearchProvider,
    model: BaseChatModel,
    scraper: IScraper,
    ranker: IRanker,
    embedder: IEmbedder | None = None,
    session=None,
) -> AsyncGenerator[PipelineEvent]:
    """Run the search pipeline, yielding (event, payload) pairs.

    When an embedder and DB session are provided, scraped chunks are
    embedded into the vector store and context is retrieved via hybrid
    retrieval (vector candidates -> cross-encoder rerank).

    Errors are reported as ``error`` events, never raised, so streaming
    consumers always get a well-formed termination.
    """
    try:
        chunk_repo = ChunkRepository(session) if session is not None else None
        toolbox = AskToolbox(searching, scraper, ranker, embedder, chunk_repo)

        # <----- Query rewriting (native tool call) ----->
        yield "status", {"step": "Building query for search engine..."}
        async with _stage("query_optimization"):
            optimized_query, used_tool_call = await rewrite_query(model, query)

        yield "query", {"query": optimized_query, "tool_call": used_tool_call}
        if settings.DEBUG:
            yield "debug", {
                "stage": "query_optimization",
                "used_tool_call": used_tool_call,
            }

        # <----- Web search + ranking ----->
        yield "status", {"step": "Searching URLs..."}
        async with _stage("web_search"):
            await toolbox.web_search(optimized_query)

        if not toolbox.ranked_sources:
            yield "error", {"error": "No search results found."}
            return

        yield "sources", {"sources": toolbox.ranked_sources}

        # <----- Scraping + chunking ----->
        yield "status", {"step": "Downloading data..."}
        async with _stage("fetch_pages"):
            await toolbox.fetch_pages()

        # <----- Context selection ----->
        yield "status", {"step": "Selecting context..."}
        async with _stage("get_context"):
            context = await toolbox.get_context(query)

        yield "context", {
            "context": context,
            "chunks": len(toolbox.selected_chunks),
            "vector_retrieval": toolbox.used_vector_retrieval,
        }

        yield (
            "answer_ready",
            {
                "context": context,
                "sources": toolbox.ranked_sources,
                "query": query,
                "optimized_query": optimized_query,
            },
        )

    except Exception as e:
        logger.exception("Search pipeline failed for query: %s", query)
        yield "error", {"error": f"Critical error: {str(e)}"}


def _sse_frame(event_id: int, event: str, data: dict[str, Any]) -> str:
    """One Server-Sent Events frame with id, event, and terminator."""
    return f"event: {event}\nid: {event_id}\ndata: {json.dumps(data)}\n\n"


async def _stream_answer_frames(query: str, context: str, model: BaseChatModel):
    """Yield answer text deltas; deterministic text when context is empty."""
    if not context.strip():
        yield NO_CONTEXT_ANSWER
        return

    async for delta in stream_cited_answer(model, query, context):
        yield delta


# <----- SSE Adapter ----->
async def search_sse_adapter(
    query: str,
    searching: ISearchProvider,
    model: BaseChatModel,
    scraper: IScraper,
    ranker: IRanker,
    embedder: IEmbedder | None = None,
    session=None,
) -> AsyncGenerator[str]:
    """Translate pipeline events into an SSE stream (protocol v2).

    Once streaming has started, errors MUST be emitted as ``error`` events;
    raising HTTPException here would kill the stream mid-flight.
    """
    event_id = 0

    def frame(event: str, data: dict[str, Any]) -> str:
        nonlocal event_id
        event_id += 1
        return _sse_frame(event_id, event, data)

    pipeline = search_logic(
        query, searching, model, scraper, ranker, embedder=embedder, session=session
    )
    async for event, data in pipeline:
        if event == "answer_ready":
            yield frame("status", {"step": "Generating answer..."})

            try:
                async for delta in _stream_answer_frames(
                    data["query"], data["context"], model
                ):
                    yield frame("message", {"text": delta})

                if not data["context"].strip():
                    yield frame("done", {"status": "no_context"})
                else:
                    yield frame("done", {"status": "completed"})
            except ToolProtocolError as e:
                logger.exception("Answer streaming failed for query: %s", query)
                yield frame("error", {"error": str(e)})
            except Exception as e:
                logger.exception("Answer streaming failed for query: %s", query)
                yield frame("error", {"error": f"Answer generation failed: {str(e)}"})
            return
        else:
            yield frame(event, data)


# <----- JSON Adapter ----->
async def search_json_adapter(
    query: str,
    searching: ISearchProvider,
    model: BaseChatModel,
    scraper: IScraper,
    ranker: IRanker,
    embedder: IEmbedder | None = None,
    session=None,
) -> dict[str, Any]:
    """Run the pipeline and return the full result as one JSON-ready dict."""
    answer_data: dict[str, Any] | None = None
    error_detail: str | None = None

    pipeline = search_logic(
        query, searching, model, scraper, ranker, embedder=embedder, session=session
    )
    async for event, data in pipeline:
        if event == "error":
            error_detail = data.get("error", "Failed to execute search logic.")
            break
        if event == "answer_ready":
            answer_data = data
            break

    if error_detail:
        raise HTTPException(status_code=500, detail=error_detail)

    if answer_data is None:
        raise HTTPException(
            status_code=500,
            detail="No search data produced (no page content could be extracted).",
        )

    context = answer_data["context"]
    if not context.strip():
        answer = NO_CONTEXT_ANSWER
    else:
        try:
            answer = await generate_cited_answer(
                model, answer_data["query"], context
            )
        except ToolProtocolError as e:
            raise HTTPException(
                status_code=502,
                detail=f"{e} The model must support tool calling.",
            ) from e
        except Exception as e:
            logger.exception("Answer generation failed for query: %s", query)
            raise HTTPException(
                status_code=500, detail=f"Answer generation failed: {str(e)}"
            ) from e

    return {
        "query": answer_data["query"],
        "optimized_query": answer_data["optimized_query"],
        "sources": answer_data["sources"],
        "context": context,
        "answer": answer,
    }
