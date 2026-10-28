"""Research pipeline: planner -> per-sub-question search -> cited report.

`research_logic` yields (event, payload) pairs; adapters translate them
into SSE frames (protocol v2 extensions, see docs/api/) or one JSON
document — the same shape as the ask pipeline adapters.
"""

import json
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import nullcontext
from typing import Any

from fastapi import HTTPException, status
from langchain_core.language_models import BaseChatModel

from ....modules.chunk_store.repository import ChunkRepository
from ....modules.embedding.base import IEmbedder
from ....modules.jobs.gates import (
    GateBusy,
    TurnResources,
    provider_busy_detail,
)
from ....modules.owl_reranker.base import IRanker
from ....modules.scrapy.providers.base import IScraper
from ....modules.search_engine.providers.base import ISearchProvider
from ....tools.settings import settings
from ...tools.errors import ToolProtocolError
from ...tools.toolbox import AskToolbox
from .agent import agent_logic
from .budgets import Budget
from .planner import plan_subquestions
from .synthesis import NO_CONTEXT_REPORT, generate_report, stream_report

logger = logging.getLogger(__name__)

PipelineEvent = tuple[str, dict[str, Any]]


def _bounded(overrides: dict, key: str, ceiling: int | float, cast) -> int | float:
    """Persona override for `key`, cast and clamped to [1, ceiling].

    Malformed values (wrong type, unparseable) fall back to the ceiling
    (the settings value); personas are user input and never crash a run.
    """
    raw = overrides.get(key, ceiling)
    try:
        value = cast(raw)
    except (TypeError, ValueError):
        return ceiling
    return max(1, min(value, ceiling))


def make_budget(overrides: dict | None = None) -> Budget:
    """Budget for one research/agent run, from settings.

    A persona's `default_parameters` may override the bounded subset
    (max_iterations, time_budget_seconds, token_budget); unknown keys are
    ignored and the settings values always act as the ceiling.
    """
    overrides = overrides or {}
    return Budget(
        max_iterations=int(
            _bounded(overrides, "max_iterations", settings.AGENT_MAX_ITERATIONS, int)
        ),
        time_budget_seconds=float(
            _bounded(
                overrides,
                "time_budget_seconds",
                settings.AGENT_TIME_BUDGET_SECONDS,
                float,
            )
        ),
        token_budget=int(
            _bounded(overrides, "token_budget", settings.AGENT_TOKEN_BUDGET, int)
        ),
    )


async def research_logic(
    query: str,
    searching: ISearchProvider,
    model: BaseChatModel,
    scraper: IScraper,
    ranker: IRanker,
    embedder: IEmbedder | None = None,
    session=None,
    today: str | None = None,
    source_types: list[str] | None = None,
    attachments: list[dict[str, Any]] | None = None,
) -> AsyncGenerator[PipelineEvent]:
    """Run the research pipeline, yielding (event, payload) pairs.

    Planning and synthesis are LLM tool calls; per-sub-question gathering
    reuses the shared pipeline tools (web_search -> fetch_pages ->
    get_context) exactly like the search pipeline, minus query rewriting
    (the planner already produced search-shaped sub-questions).

    Errors are reported as ``error`` events, never raised.
    """
    budget = make_budget()
    try:
        chunk_repo = ChunkRepository(session) if session is not None else None
        toolbox = AskToolbox(
            searching, scraper, ranker, embedder, chunk_repo,
            source_types=source_types,
            attachments=attachments,
        )

        # <----- Planning (native tool call) ----->
        yield "status", {"step": "Planning sub-questions..."}
        subquestions, used_tool_call = await plan_subquestions(
            model, query, settings.RESEARCH_MAX_SUBQUESTIONS, today=today
        )
        budget.spend_tokens(" ".join(subquestions))

        yield "plan", {"subquestions": subquestions, "tool_call": used_tool_call}
        if settings.DEBUG:
            yield "debug", {"stage": "planning", "used_tool_call": used_tool_call}

        # <----- Per-sub-question gathering (shared tools) ----->
        gathered: list[dict[str, Any]] = []
        for index, subquestion in enumerate(subquestions, start=1):
            if budget.exhausted:
                reason = budget.stop_reason() or "budget exhausted"
                yield "status", {
                    "step": f"Skipping remaining sub-questions: {reason}."
                }
                break

            yield "status", {
                "step": f"Researching {index}/{len(subquestions)}: {subquestion}"
            }
            start = time.perf_counter()

            await toolbox.web_search(subquestion)
            if toolbox.ranked_sources:
                yield "sources", {
                    "sources": toolbox.ranked_sources,
                    "subquery": subquestion,
                }

            await toolbox.fetch_pages()
            context = await toolbox.get_context(subquestion)
            budget.spend_tokens(context)

            yield "context", {
                "context": context,
                "chunks": len(toolbox.selected_chunks),
                "vector_retrieval": toolbox.used_vector_retrieval,
                "subquery": subquestion,
            }
            if settings.DEBUG:
                yield "debug", {
                    "stage": "subquestion",
                    "subquery": subquestion,
                    "seconds": round(time.perf_counter() - start, 3),
                }

            if context.strip():
                gathered.append(
                    {
                        "subquestion": subquestion,
                        "sources": list(toolbox.ranked_sources),
                        "context": context,
                    }
                )

        if not gathered:
            yield (
                "answer_ready",
                {
                    "query": query,
                    "gathered": [],
                    "no_context": True,
                    "subquestions": subquestions,
                },
            )
            return

        yield (
            "answer_ready",
            {
                "query": query,
                "gathered": gathered,
                "no_context": False,
                "subquestions": subquestions,
            },
        )

    except Exception as e:
        logger.exception("Research pipeline failed for query: %s", query)
        yield "error", {"error": f"Critical error: {str(e)}"}


async def research_sse_adapter(
    query: str,
    searching: ISearchProvider,
    model: BaseChatModel,
    scraper: IScraper,
    ranker: IRanker,
    embedder: IEmbedder | None = None,
    session=None,
    today: str | None = None,
    source_types: list[str] | None = None,
    attachments: list[dict[str, Any]] | None = None,
    turn: TurnResources | None = None,
) -> AsyncGenerator[str]:
    """Translate research events into an SSE stream.

    Once streaming has started, errors MUST be emitted as ``error`` events;
    raising HTTPException here would kill the stream mid-flight.
    """
    event_id = 0

    def frame(event: str, data: dict[str, Any]) -> str:
        nonlocal event_id
        event_id += 1
        return f"event: {event}\nid: {event_id}\ndata: {json.dumps(data)}\n\n"

    try:
        async with turn.gated() if turn is not None else nullcontext():
            pipeline = research_logic(
                query, searching, model, scraper, ranker, embedder=embedder,
                session=session, today=today, source_types=source_types,
                attachments=attachments,
            )
            async for event, data in pipeline:
                if event == "answer_ready":
                    yield frame("status", {"step": "Writing research report..."})

                    if data["no_context"]:
                        yield frame("message", {"text": NO_CONTEXT_REPORT})
                        yield frame("done", {"status": "no_context"})
                        return

                    try:
                        report = stream_report(
                            model, data["query"], data["gathered"], today=today
                        )
                        async for delta in report:
                            yield frame("message", {"text": delta})
                        yield frame("done", {"status": "completed"})
                    except ToolProtocolError as e:
                        logger.exception(
                            "Report streaming failed for query: %s", query
                        )
                        yield frame("error", {"error": str(e)})
                    except Exception as e:
                        logger.exception(
                            "Report streaming failed for query: %s", query
                        )
                        yield frame(
                            "error",
                            {"error": f"Report generation failed: {str(e)}"},
                        )
                    return
                yield frame(event, data)
    except GateBusy:
        logger.info("Research turn rejected: resource gates busy (query: %s).", query)
        yield frame("error", provider_busy_detail())
        return


async def agent_sse_adapter(
    goal: str,
    model: BaseChatModel,
    toolbox: AskToolbox,
    budget: Budget,
    persona=None,
    turn: TurnResources | None = None,
) -> AsyncGenerator[str]:
    """Translate agent-loop events into an SSE stream.

    `turn` (PR 12) gates the whole loop; a busy gate past the bounded
    wait surfaces as a coded ``error`` event.
    """
    event_id = 0

    def frame(event: str, data: dict[str, Any]) -> str:
        nonlocal event_id
        event_id += 1
        return f"event: {event}\nid: {event_id}\ndata: {json.dumps(data)}\n\n"

    try:
        async with turn.gated() if turn is not None else nullcontext():
            async for event, data in agent_logic(
                goal, model, toolbox, budget, persona
            ):
                yield frame(event, data)
    except GateBusy:
        logger.info("Agent turn rejected: resource gates busy (goal: %s).", goal)
        yield frame("error", provider_busy_detail())
        return


async def research_json_adapter(
    query: str,
    searching: ISearchProvider,
    model: BaseChatModel,
    scraper: IScraper,
    ranker: IRanker,
    embedder: IEmbedder | None = None,
    session=None,
    today: str | None = None,
    source_types: list[str] | None = None,
    attachments: list[dict[str, Any]] | None = None,
    turn: TurnResources | None = None,
) -> dict[str, Any]:
    """Run the research pipeline and return the full result as one dict."""
    answer_data: dict[str, Any] | None = None
    error_detail: str | None = None

    try:
        async with turn.gated() if turn is not None else nullcontext():
            pipeline = research_logic(
                query, searching, model, scraper, ranker, embedder=embedder,
                session=session, today=today, source_types=source_types,
                attachments=attachments,
            )
            async for event, data in pipeline:
                if event == "error":
                    error_detail = data.get(
                        "error", "Failed to execute research logic."
                    )
                    break
                if event == "answer_ready":
                    answer_data = data
                    break

            if error_detail:
                raise HTTPException(status_code=500, detail=error_detail)

            if answer_data is None:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "No research data produced "
                        "(no page content could be extracted)."
                    ),
                )

            if answer_data["no_context"]:
                report = NO_CONTEXT_REPORT
            else:
                try:
                    report = await generate_report(
                        model, answer_data["query"], answer_data["gathered"], today
                    )
                except ToolProtocolError as e:
                    raise HTTPException(
                        status_code=502,
                        detail=f"{e} The model must support tool calling.",
                    ) from e
                except Exception as e:
                    logger.exception("Report generation failed for query: %s", query)
                    raise HTTPException(
                        status_code=500,
                        detail=f"Report generation failed: {str(e)}",
                    ) from e

            return {
                "query": answer_data["query"],
                "subquestions": answer_data["subquestions"],
                "gathered": answer_data["gathered"],
                "report": report,
            }
    except GateBusy as e:
        logger.info("Research turn rejected: resource gates busy (query: %s).", query)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=provider_busy_detail(),
        ) from e
