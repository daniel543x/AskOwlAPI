"""Research pipeline: planner -> per-sub-question search -> cited report.

`research_logic` yields (event, payload) pairs; adapters translate them
into SSE frames (protocol v2 extensions, see docs/api/) or one JSON
document — the same shape as the ask pipeline adapters.
"""

import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import HTTPException
from langchain_core.language_models import BaseChatModel

from ....modules.chunk_store.repository import ChunkRepository
from ....modules.embedding.base import IEmbedder
from ....modules.owl_reranker.base import IRanker
from ....modules.scrapy.providers.base import IScraper
from ....modules.search_engine.providers.base import ISearchProvider
from ....tools.settings import settings
from ...tools.errors import ToolProtocolError
from ...tools.toolbox import AskToolbox
from .agent import agent_logic
from .budgets import Budget
from .compression import compress
from .planner import plan_subquestions
from .reflection import reflect
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
    max_iterations: int = 1,
    today: str | None = None,
) -> AsyncGenerator[PipelineEvent]:
    """Run the research pipeline, yielding (event, payload) pairs.

    Planning and synthesis are LLM tool calls; per-sub-question gathering
    reuses the shared pipeline tools (web_search -> fetch_pages ->
    get_context) exactly like the search pipeline, minus query rewriting
    (the planner already produced search-shaped sub-questions).

    WU-7: with `max_iterations > 1` (clamped to RESEARCH_MAX_ROUNDS), a
    reflect -> follow-up-search loop runs after the first pass; WU-8
    compresses each sub-question's context into a running brief when
    `RESEARCH_COMPRESS_CONTEXT` is on. Defaults reproduce the single-pass,
    raw-context pipeline exactly.

    Errors are reported as ``error`` events, never raised.
    """
    budget = make_budget()
    rounds_allowed = max(1, min(max_iterations, settings.RESEARCH_MAX_ROUNDS))
    try:
        chunk_repo = ChunkRepository(session) if session is not None else None
        toolbox = AskToolbox(searching, scraper, ranker, embedder, chunk_repo)

        # <----- Planning (native tool call) ----->
        yield "status", {"step": "Planning sub-questions..."}
        subquestions, used_tool_call = await plan_subquestions(
            model, query, settings.RESEARCH_MAX_SUBQUESTIONS, today=today
        )
        budget.spend_tokens(" ".join(subquestions))

        yield "plan", {"subquestions": subquestions, "tool_call": used_tool_call}
        if settings.DEBUG:
            yield "debug", {"stage": "planning", "used_tool_call": used_tool_call}

        gathered: list[dict[str, Any]] = []
        briefs: dict[str, str] = {}
        researched: list[str] = []

        async def gather(subquestion: str) -> AsyncGenerator[PipelineEvent]:
            """Search + absorb one query into `gathered` (brief when on)."""
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

            if settings.RESEARCH_COMPRESS_CONTEXT:
                try:
                    context = await compress(
                        model,
                        subquestion,
                        briefs.get(subquestion, ""),
                        context,
                        settings.SUBQUESTION_BRIEF_TOKENS,
                    )
                except Exception:
                    logger.warning(
                        "Context compression failed; keeping raw context.",
                        exc_info=True,
                    )
                briefs[subquestion] = context
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
            researched.append(subquestion)

        # <----- Per-sub-question gathering (shared tools) ----->
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
            async for event_name, data in gather(subquestion):
                yield event_name, data

        # <----- Reflect -> follow-up loop (WU-7) ----->
        # rounds_allowed counts total gathering passes: pass 1 is the
        # planner pass, so reflection cycles = rounds_allowed - 1.
        rounds_used = 0
        if rounds_allowed > 1 and gathered:
            for round_no in range(1, rounds_allowed):
                if budget.exhausted:
                    break
                missing_info = await reflect(
                    model,
                    query,
                    [
                        {"subquestion": g["subquestion"], "context": g["context"]}
                        for g in gathered
                    ],
                )
                budget.spend_iteration()
                rounds_used = round_no
                yield "reflect", {
                    "missing": missing_info.missing,
                    "follow_ups": missing_info.follow_up_queries,
                    "rounds_used": rounds_used,
                }
                if settings.DEBUG:
                    yield "debug", {
                        "stage": "reflection",
                        "round": round_no,
                        "missing": len(missing_info.missing),
                    }

                follow_ups = [
                    q
                    for q in missing_info.follow_up_queries
                    if q.strip() and q.strip() not in researched
                ]
                if not follow_ups:
                    break

                for follow_up in follow_ups:
                    if budget.exhausted:
                        break
                    yield "status", {
                        "step": f"Follow-up (round {round_no}): {follow_up}"
                    }
                    async for event_name, data in gather(follow_up):
                        yield event_name, data

        stop_reason = budget.stop_reason() if budget.exhausted else None

        if not gathered:
            yield (
                "answer_ready",
                {
                    "query": query,
                    "gathered": [],
                    "no_context": True,
                    "subquestions": subquestions,
                    "rounds_used": rounds_used,
                    "budget_stop_reason": None,
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
                "rounds_used": rounds_used,
                "budget_stop_reason": stop_reason,
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
    max_iterations: int = 1,
    today: str | None = None,
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

    pipeline = research_logic(
        query, searching, model, scraper, ranker, embedder=embedder, session=session,
        max_iterations=max_iterations, today=today,
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
                if data.get("budget_stop_reason"):
                    yield frame(
                        "done",
                        {
                            "status": "budget_exhausted",
                            "reason": data["budget_stop_reason"],
                        },
                    )
                else:
                    yield frame("done", {"status": "completed"})
            except ToolProtocolError as e:
                logger.exception("Report streaming failed for query: %s", query)
                yield frame("error", {"error": str(e)})
            except Exception as e:
                logger.exception("Report streaming failed for query: %s", query)
                yield frame("error", {"error": f"Report generation failed: {str(e)}"})
            return
        yield frame(event, data)


async def agent_sse_adapter(
    goal: str,
    model: BaseChatModel,
    toolbox: AskToolbox,
    budget: Budget,
    persona=None,
) -> AsyncGenerator[str]:
    """Translate agent-loop events into an SSE stream."""
    event_id = 0

    def frame(event: str, data: dict[str, Any]) -> str:
        nonlocal event_id
        event_id += 1
        return f"event: {event}\nid: {event_id}\ndata: {json.dumps(data)}\n\n"

    async for event, data in agent_logic(goal, model, toolbox, budget, persona):
        yield frame(event, data)


async def research_json_adapter(
    query: str,
    searching: ISearchProvider,
    model: BaseChatModel,
    scraper: IScraper,
    ranker: IRanker,
    embedder: IEmbedder | None = None,
    session=None,
    max_iterations: int = 1,
    today: str | None = None,
) -> dict[str, Any]:
    """Run the research pipeline and return the full result as one dict."""
    answer_data: dict[str, Any] | None = None
    error_detail: str | None = None

    pipeline = research_logic(
        query, searching, model, scraper, ranker, embedder=embedder, session=session,
        max_iterations=max_iterations, today=today,
    )
    async for event, data in pipeline:
        if event == "error":
            error_detail = data.get("error", "Failed to execute research logic.")
            break
        if event == "answer_ready":
            answer_data = data
            break

    if error_detail:
        raise HTTPException(status_code=500, detail=error_detail)

    if answer_data is None:
        raise HTTPException(
            status_code=500,
            detail="No research data produced (no page content could be extracted).",
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
                status_code=500, detail=f"Report generation failed: {str(e)}"
            ) from e

    return {
        "query": answer_data["query"],
        "subquestions": answer_data["subquestions"],
        "gathered": answer_data["gathered"],
        "rounds_used": answer_data["rounds_used"],
        "budget_stop_reason": answer_data["budget_stop_reason"],
        "report": report,
    }
