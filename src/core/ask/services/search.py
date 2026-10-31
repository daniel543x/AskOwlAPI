"""Ask pipeline (protocol v2): every LLM step goes through a tool call.

`search_logic` runs the linear pipeline (rewrite -> web_search ->
fetch_pages -> get_context) and yields (event, payload) pairs. Adapters
translate the events into SSE (with event ids) or one JSON document.
"""

import json
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, nullcontext
from typing import Any

from fastapi import HTTPException, status
from langchain_core.language_models import BaseChatModel

from ....modules.chunk_store.repository import ChunkRepository
from ....modules.embedding.base import IEmbedder
from ....modules.history.model import Conversation
from ....modules.history.repository import HistoryRepository
from ....modules.jobs.gates import (
    GateBusy,
    TurnResources,
    provider_busy_detail,
)
from ....modules.owl_reranker.base import IRanker
from ....modules.scrapy.providers.base import IScraper
from ....modules.search_engine.providers.base import ISearchProvider
from ....tools.cache_flags import cache_bypass
from ....tools.settings import settings
from ...tools.citations import CitationRewriter
from ...tools.pre_scores import choose_query, grounding_confidence
from ...tools.toolbox import AskToolbox
from .llm_steps import (
    ToolProtocolError,
    generate_cited_answer,
    rewrite_query,
    stream_cited_answer,
)
from .variations import generate_variations

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
    decision_maker=None,
    max_urls: int | None = None,
    query_variations: int = 1,
    scope: str = "global",
    recency_bias: float = 0.0,
    conversation_id=None,
    today: str | None = None,
    source_types: list[str] | None = None,
    attachments: list[dict[str, Any]] | None = None,
    history: list[tuple[str, str]] | None = None,
    rewrite: str = "standalone",
    hybrid_alpha: float = 1.0,
) -> AsyncGenerator[PipelineEvent]:
    """Run the search pipeline, yielding (event, payload) pairs.

    When an embedder and DB session are provided, scraped chunks are
    embedded into the vector store and context is retrieved via hybrid
    retrieval (vector candidates -> cross-encoder rerank).

    Errors are reported as ``error`` events, never raised, so streaming
    consumers always get a well-formed termination.

    With `AGENTIC_SEARCH` on (PR 17), the turn runs the agentic core
    first; a tool-contract violation or a budget exhaustion without
    material emits a ``degraded`` event and falls back to this
    deterministic body (decision 8).
    """
    if settings.AGENTIC_SEARCH:
        try:
            from ..agentic import (
                AgentDegraded,
                agentic_search_logic,
                search_budget,
            )

            pre_toolbox = AskToolbox(
                searching, scraper, ranker, embedder, None,
                decision_maker=decision_maker,
                max_urls=max_urls,
                attachments=attachments,
                hybrid_alpha=hybrid_alpha,
            )
            async for event, data in agentic_search_logic(
                query, model, pre_toolbox, search_budget(), history
            ):
                yield event, data
            return
        except AgentDegraded as e:
            logger.warning("Agentic search degraded: %s", e)
            yield "degraded", {"reason": str(e)}
    try:
        chunk_repo = ChunkRepository(session) if session is not None else None
        toolbox = AskToolbox(
            searching, scraper, ranker, embedder, chunk_repo,
            decision_maker=decision_maker,
            max_urls=max_urls,
            conversation_id=conversation_id if scope == "conversation" else None,
            scope=scope,
            recency_bias=recency_bias,
            source_types=source_types,
        )

        # <----- Query rewriting (native tool call, WU-1) ----->
        yield "status", {"step": "Building query for search engine..."}
        if rewrite == "none":
            # The permanent escape hatch (eval/tool callers): skip the
            # rewriting LLM call entirely.
            optimized_query, used_tool_call, standalone = query, False, False
        else:
            history_for_rewrite = history if rewrite == "standalone" else None
            async with _stage("query_optimization"):
                (
                    optimized_query,
                    used_tool_call,
                    standalone,
                ) = await rewrite_query(
                    model, query, history_for_rewrite, today=today
                )

        # <----- Query-choice pre-score (optional decision layer) ----->
        if settings.PRE_SCORE_QUERY_CHOICE and decision_maker:
            async with _stage("query_choice"):
                optimized_query = await choose_query(
                    decision_maker, query, optimized_query
                )

        yield "query", {
            "query": optimized_query,
            "tool_call": used_tool_call,
            "standalone": standalone,
        }
        if settings.DEBUG:
            yield "debug", {
                "stage": "query_optimization",
                "used_tool_call": used_tool_call,
            }

        # <----- Web search + ranking (optional rule-based fan-out, WU-2) ----->
        variation_count = max(1, min(query_variations, settings.QUERY_VARIATIONS_MAX))
        queries = generate_variations(optimized_query, variation_count)
        yield "status", {"step": "Searching URLs..."}
        async with _stage("web_search"):
            if len(queries) > 1:
                yield "status", {
                    "step": f"Searching {len(queries)} query variants..."
                }
                await toolbox.web_search_many(queries)
            else:
                await toolbox.web_search(optimized_query)

        if not toolbox.ranked_sources:
            yield "error", {"error": "No search results found."}
            return

        yield "sources", {"sources": toolbox.ranked_sources}
        if settings.DEBUG:
            yield "debug", {"stage": "web_search", **toolbox.search_stats}

        # <----- Scraping + chunking ----->
        yield "status", {"step": "Downloading data..."}
        async with _stage("fetch_pages"):
            await toolbox.fetch_pages()

        # <----- Context selection ----->
        yield "status", {"step": "Selecting context..."}
        async with _stage("get_context"):
            context = await toolbox.get_context(query)

        if attachments:
            from sqlalchemy.ext.asyncio import AsyncSession as _AS

            from ....modules.attachments.service import attachment_context
            from ....tools.db import engine as _engine

            # Deterministic injection (PR 15): uploaded files join the
            # context directly — the search mode has no tool loop.
            async with _AS(_engine, expire_on_commit=False) as att_session:
                file_context = await attachment_context(att_session, attachments)
            if file_context:
                context = (
                    f"{file_context}\n\n{context}" if context else file_context
                )
                yield "status", {"step": "Added uploaded files to context."}

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


async def _stream_answer_frames(
    query: str,
    context: str,
    model: BaseChatModel,
    history: list[tuple[str, str]] | None = None,
    today: str | None = None,
    images: list[bytes] | None = None,
):
    """Yield ("answer"|"thinking", text) deltas; deterministic text when
    context is empty."""
    if not context.strip():
        yield "answer", NO_CONTEXT_ANSWER
        return

    async for kind, delta in stream_cited_answer(
        model, query, context, history, today, images
    ):
        yield kind, delta


async def _resolve_conversation(
    session, user_id, conversation_id, query: str
) -> tuple[Conversation | None, list[tuple[str, str]], bool]:
    """Load (or create) the caller's conversation and its prior turns.

    Returns (conversation, history, created). Best-effort: when persistence
    is unavailable (no session/user) the pipeline runs stateless.
    """
    if session is None or user_id is None:
        return None, [], False

    repo = HistoryRepository(session)
    if conversation_id is not None:
        conversation = await repo.get_conversation_for_user(user_id, conversation_id)
        if conversation is None:
            raise KeyError("Conversation not found.")
        created = False
    else:
        conversation = await repo.create_conversation(user_id, title=query)
        created = True

    messages = await repo.get_messages(conversation.id)
    history = [(message.role, message.content) for message in messages]
    return conversation, history, created


async def _save_turn(session, conversation, query: str, answer: str) -> None:
    if conversation is None:
        return
    try:
        await HistoryRepository(session).add_turn(conversation, query, answer)
    except Exception:
        # Persistence must never kill a finished answer stream.
        logger.exception("Failed to persist conversation turn.")


# <----- SSE Adapter ----->
async def search_sse_adapter(
    query: str,
    searching: ISearchProvider,
    model: BaseChatModel,
    scraper: IScraper,
    ranker: IRanker,
    embedder: IEmbedder | None = None,
    session=None,
    user_id=None,
    conversation_id=None,
    decision_maker=None,
    max_urls: int | None = None,
    query_variations: int = 1,
    cache: str = "on",
    scope: str = "global",
    recency_bias: float = 0.0,
    today: str | None = None,
    source_types: list[str] | None = None,
    attachments: list[dict[str, Any]] | None = None,
    images: list[bytes] | None = None,
    rewrite: str = "standalone",
    hybrid_alpha: float = 1.0,
    turn: TurnResources | None = None,
) -> AsyncGenerator[str]:
    """Translate pipeline events into an SSE stream (protocol v2).

    Once streaming has started, errors MUST be emitted as ``error`` events;
    raising HTTPException here would kill the stream mid-flight.

    With `session` and `user_id`, the turn is persisted to the caller's
    conversation (`conversation_id`, or a new conversation when omitted);
    prior turns are injected as chat history into the answer step.

    `turn` (PR 12) holds the resource gates for the whole stream: the
    first frame is delayed until the gates free up, and a busy gate that
    outlasts the bounded wait surfaces as an ``error`` event with the
    coded ``provider_busy`` body.
    """
    event_id = 0

    def frame(event: str, data: dict[str, Any]) -> str:
        nonlocal event_id
        event_id += 1
        return _sse_frame(event_id, event, data)

    async def _sse_body():
        try:
            conversation, history, created = await _resolve_conversation(
                session, user_id, conversation_id, query
            )
        except KeyError:
            yield frame("error", {"error": "Conversation not found."})
            return
        if conversation is not None:
            yield frame(
                "conversation",
                {"conversation_id": str(conversation.id), "created": created},
            )

        bypass = cache == "bypass"
        with cache_bypass() if bypass else nullcontext():
            async for event, data in search_logic(
                query, searching, model, scraper, ranker, embedder=embedder,
                session=session, decision_maker=decision_maker, max_urls=max_urls,
                query_variations=query_variations,
                scope=scope, recency_bias=recency_bias,
                conversation_id=conversation.id if conversation is not None else None,
                today=today,
                source_types=source_types,
                attachments=attachments,
                history=history,
                rewrite=rewrite,
                hybrid_alpha=hybrid_alpha,
            ):
                if event == "answer_ready":
                    yield frame("status", {"step": "Generating answer..."})

                    known = [
                        s["url"] for s in data["sources"] if s.get("url")
                    ]
                    rewriter = CitationRewriter(known)
                    answer_parts: list[str] = []
                    try:
                        async for kind, delta in _stream_answer_frames(
                            data["query"], data["context"], model, history,
                            today, images,
                        ):
                            if kind == "thinking":
                                yield frame("thinking", {"text": delta})
                                continue
                            text = rewriter.feed(delta)
                            if not text:
                                continue
                            answer_parts.append(text)
                            yield frame("message", {"text": text})

                        tail = rewriter.flush()
                        if tail:
                            answer_parts.append(tail)
                            yield frame("message", {"text": tail})
                        answer_text = "".join(answer_parts)
                        citations = rewriter.citations_payload()
                        if citations:
                            yield frame("citations", {"citations": citations})
                        await _save_turn(
                            session, conversation, data["query"], answer_text
                        )

                        if settings.PRE_SCORE_GROUNDING_CHECK and decision_maker:
                            supported = await grounding_confidence(
                                decision_maker, data["query"], data["context"],
                                answer_text,
                            )
                            if supported is not None:
                                yield frame(
                                    "grounding", {"supported": round(supported, 4)}
                                )

                        if not data["context"].strip():
                            yield frame("done", {"status": "no_context"})
                        else:
                            yield frame("done", {"status": "completed"})
                    except ToolProtocolError as e:
                        logger.exception(
                            "Answer streaming failed for query: %s", query
                        )
                        yield frame("error", {"error": str(e)})
                    except Exception as e:
                        logger.exception(
                            "Answer streaming failed for query: %s", query
                        )
                        yield frame(
                            "error", {"error": f"Answer generation failed: {str(e)}"}
                        )
                    return
                elif event == "agentic_answer":
                    # PR 17: the core already streamed the answer deltas;
                    # persist the turn and terminate.
                    answer_text = data["answer"]
                    await _save_turn(session, conversation, data["query"], answer_text)
                    if not data["context"].strip():
                        yield frame("done", {"status": "no_context"})
                    else:
                        yield frame("done", {"status": "completed"})
                    return
                else:
                    yield frame(event, data)

    try:
        async with turn.gated() if turn is not None else nullcontext():
            async for item in _sse_body():
                yield item
    except GateBusy:
        logger.info("Search turn rejected: resource gates busy (query: %s).", query)
        yield frame("error", provider_busy_detail())
        return


# <----- JSON Adapter ----->
async def search_json_adapter(
    query: str,
    searching: ISearchProvider,
    model: BaseChatModel,
    scraper: IScraper,
    ranker: IRanker,
    embedder: IEmbedder | None = None,
    session=None,
    user_id=None,
    conversation_id=None,
    decision_maker=None,
    max_urls: int | None = None,
    query_variations: int = 1,
    cache: str = "on",
    scope: str = "global",
    recency_bias: float = 0.0,
    today: str | None = None,
    source_types: list[str] | None = None,
    attachments: list[dict[str, Any]] | None = None,
    images: list[bytes] | None = None,
    rewrite: str = "standalone",
    hybrid_alpha: float = 1.0,
    turn: TurnResources | None = None,
) -> dict[str, Any]:
    """Run the pipeline and return the full result as one JSON-ready dict.

    With `session` and `user_id` the turn is persisted to a conversation
    (see `search_sse_adapter`); an unknown `conversation_id` is a 404."""
    answer_data: dict[str, Any] | None = None
    error_detail: str | None = None

    try:
        conversation, history, _created = await _resolve_conversation(
            session, user_id, conversation_id, query
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    try:
        async with turn.gated() if turn is not None else nullcontext():
            bypass = cache == "bypass"
            with cache_bypass() if bypass else nullcontext():
                pipeline = search_logic(
                    query, searching, model, scraper, ranker, embedder=embedder,
                    session=session, decision_maker=decision_maker, max_urls=max_urls,
                    query_variations=query_variations,
                    scope=scope, recency_bias=recency_bias,
                    conversation_id=(
                        conversation.id if conversation is not None else None
                    ),
                    today=today,
                    source_types=source_types,
                    attachments=attachments,
                    history=history,
                    rewrite=rewrite,
                    hybrid_alpha=hybrid_alpha,
                )
                async for event, data in pipeline:
                    if event == "error":
                        error_detail = data.get(
                            "error", "Failed to execute search logic."
                        )
                        break
                    if event == "answer_ready":
                        answer_data = data
                        break
                    if event == "agentic_answer":
                        answer_data = {**data, "direct_answer": data["answer"]}
                        break
    except GateBusy as e:
        logger.info("Search turn rejected: resource gates busy (query: %s).", query)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=provider_busy_detail(),
        ) from e

    if error_detail:
        raise HTTPException(status_code=500, detail=error_detail)

    if answer_data is None:
        raise HTTPException(
            status_code=500,
            detail="No search data produced (no page content could be extracted).",
        )

    context = answer_data["context"]
    if "direct_answer" in answer_data:
        # PR 17: the agentic core already produced (and streamed) the
        # answer; never re-generate.
        answer = answer_data["direct_answer"]
    elif not context.strip():
        answer = NO_CONTEXT_ANSWER
    else:
        try:
            answer = await generate_cited_answer(
                model, answer_data["query"], context, history, today, images
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

    await _save_turn(session, conversation, answer_data["query"], answer)

    known = [
        s["url"] for s in answer_data["sources"] if isinstance(s, dict) and s.get("url")
    ]
    if "citations" not in answer_data:
        # Deterministic path: the answer is unrewritten; agentic turns
        # arrive with their citations already attached.
        rewriter = CitationRewriter(known)
        answer = rewriter.feed(answer) + rewriter.flush()
        answer_data["citations"] = rewriter.citations_payload()

    grounding = None
    if settings.PRE_SCORE_GROUNDING_CHECK and decision_maker:
        grounding = await grounding_confidence(
            decision_maker, answer_data["query"], context, answer
        )

    return {
        "query": answer_data["query"],
        "optimized_query": answer_data["optimized_query"],
        "sources": answer_data["sources"],
        "citations": answer_data.get("citations", []),
        "context": context,
        "answer": answer,
        "conversation_id": str(conversation.id) if conversation else None,
        "grounding_supported": (
            round(grounding, 4) if grounding is not None else None
        ),
    }
