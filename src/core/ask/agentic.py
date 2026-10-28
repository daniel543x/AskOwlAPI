"""The agentic search core (PR 17 / decision 8).

One tool loop shared by all modes, with the deterministic pipeline kept
as the degrade path: a tool-contract violation (one repair attempt, then
degrade) or budget exhaustion without material raises `AgentDegraded`,
which the search pipeline turns into a `degraded` SSE event plus the
deterministic run — the UI stays honest about what happened.

The search preset over the core: no planner, no note tool, low budget
(1–3 iterations). Conversation history rides in the core's system
prompt. Every LLM interaction is a native tool call; the final answer
streams through the `CitedAnswer` contract exactly like the
deterministic path.
"""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ...tools.settings import settings
from ..research.services.agent import _gathered_context, _run_tool
from ..research.services.budgets import Budget
from ..tools.arg_stream import ToolArgStream
from ..tools.errors import ToolProtocolError
from ..tools.toolbox import AskToolbox, build_tools
from .services.contracts import CitedAnswer
from .services.llm_steps import extract_reasoning_delta

logger = logging.getLogger(__name__)

PipelineEvent = tuple[str, dict[str, Any]]


class AgentDegraded(Exception):
    """The agentic loop cannot proceed; the deterministic path takes over."""


AGENTIC_SEARCH_SYSTEM = """You are an autonomous search agent.
Answer the user's request using ONLY the provided tools.

Rules:
1. Plan briefly, then call `web_search` to find pages, `fetch_pages` to
   read them, and `get_context` to select the passages that answer the
   request. Repeat with refined queries when the first pass falls short.
2. Ground every claim in the tool results; never invent facts.
3. As soon as you can answer, call `CitedAnswer` exactly once with the
   final Markdown answer, citing sources inline as [Source](URL) right
   after each claim.
4. Every response MUST contain at least one tool call. Prose-only
   responses are a protocol violation.
5. Prefer answering within a few tool rounds; this is the fast search
   preset, not an open-ended research run."""

REPAIR_NOTICE = (
    "Protocol violation: your response contained no tool call. You MUST "
    "respond by calling a tool (web_search / fetch_pages / get_context / "
    "CitedAnswer). Try again now."
)

FORCED_SYNTHESIS_NOTICE = (
    "Your tool budget is exhausted. Using STRICTLY the context gathered "
    "so far, produce the best possible final answer via the CitedAnswer "
    "tool; state explicitly what could not be established."
)


def _history_block(history: list[tuple[str, str]] | None) -> str:
    if not history:
        return ""
    recent = history[-6:]
    lines = [
        f"{role.capitalize()}: {content}" for role, content in recent
    ]
    return "\n\n--- Earlier conversation ---\n" + "\n".join(lines)


async def _stream_cited_answer(
    bound, messages: list, model: BaseChatModel
) -> AsyncGenerator[PipelineEvent]:
    """Stream the CitedAnswer answer argument as message events.

    Mirrors the deterministic answer streaming (tool-arg deltas +
    reasoning-trace passthrough). Raises ToolProtocolError when the model
    never streams the CitedAnswer value.
    """
    extractor = ToolArgStream(key="answer")
    async for chunk in bound.astream(messages):
        reasoning = extract_reasoning_delta(chunk)
        if reasoning:
            yield "thinking", {"text": reasoning}
        if not isinstance(chunk, AIMessageChunk):
            continue
        for tool_chunk in chunk.tool_call_chunks or []:
            name = tool_chunk.get("name") or ""
            if name and name != "CitedAnswer":
                continue
            delta = extractor.feed(tool_chunk.get("args") or "")
            if delta:
                yield "message", {"text": delta}
    tail = extractor.finish()
    if tail:
        yield "message", {"text": tail}
    if not extractor.saw_value:
        raise ToolProtocolError(
            "The model did not stream the answer through the CitedAnswer tool."
        )


async def agentic_search_logic(
    query: str,
    model: BaseChatModel,
    toolbox: AskToolbox,
    budget: Budget,
    history: list[tuple[str, str]] | None = None,
) -> AsyncGenerator[PipelineEvent]:
    """The low-budget search preset over the shared tool loop."""
    tools = build_tools(toolbox)
    bound = model.bind_tools([*tools, CitedAnswer])
    system = AGENTIC_SEARCH_SYSTEM + _history_block(history)
    messages: list = [SystemMessage(content=system), HumanMessage(content=query)]
    repair_attempts_left = max(0, settings.AGENTIC_REPAIR_ATTEMPTS)

    for _ in range(max(1, budget.max_iterations)):
        # Every round streams: a CitedAnswer in the same pass is answered
        # in place (one LLM call per round, deltas live), and a tool-call
        # round accumulates the message for the loop transcript.
        from langchain_core.messages import AIMessage

        extractor = ToolArgStream(key="answer")
        merged: AIMessageChunk | None = None
        streamed: list[str] = []
        async for chunk in bound.astream(messages):
            reasoning = extract_reasoning_delta(chunk)
            if reasoning:
                yield "thinking", {"text": reasoning}
            if not isinstance(chunk, AIMessageChunk):
                continue
            merged = chunk if merged is None else merged + chunk
            for tool_chunk in chunk.tool_call_chunks or []:
                name = tool_chunk.get("name") or ""
                if name and name != "CitedAnswer":
                    continue
                delta = extractor.feed(tool_chunk.get("args") or "")
                if delta:
                    streamed.append(delta)
                    yield "message", {"text": delta}
        if merged is None:
            raise AgentDegraded("the model produced an empty response")
        answer_streamed = extractor.saw_value
        if answer_streamed:
            yield "status", {"step": "Generating answer..."}
            tail = extractor.finish()
            if tail:
                yield "message", {"text": tail}
            context = await _gathered_context(toolbox)
            yield (
                "agentic_answer",
                {
                    "query": query,
                    "optimized_query": query,
                    "context": context,
                    "sources": toolbox.ranked_sources,
                    "answer": "".join(streamed),
                },
            )
            return

        response = AIMessage(
            content=merged.content, tool_calls=merged.tool_calls
        )
        budget.spend_tokens(str(response.content), str(response.tool_calls))
        executable = [
            c for c in (response.tool_calls or [])
            if c.get("name") != "CitedAnswer"
        ]
        if not executable:
            if repair_attempts_left <= 0:
                raise AgentDegraded(
                    "tool-contract violation: the model answered without "
                    "a tool call"
                )
            repair_attempts_left -= 1
            logger.warning(
                "Agentic search: contract violation; repair attempt %d.",
                settings.AGENTIC_REPAIR_ATTEMPTS - repair_attempts_left,
            )
            messages.append(response)
            messages.append(SystemMessage(content=REPAIR_NOTICE))
            continue

        messages.append(response)
        budget.spend_iteration()
        for call in executable:
            call_name = str(call.get("name", ""))
            yield "status", {"step": f"Agent tool: {call_name}(...)"}
            result = await _run_tool(toolbox, call_name, call.get("args", {}))
            messages.append(
                ToolMessage(
                    content=result, tool_call_id=str(call.get("id", ""))
                )
            )
        if toolbox.ranked_sources:
            yield "sources", {"sources": toolbox.ranked_sources}
        if settings.DEBUG:
            yield "debug", {
                "stage": "agentic_loop",
                "iterations_left": budget.iterations_left,
            }

    context = await _gathered_context(toolbox)
    if not context.strip():
        raise AgentDegraded("budget exhausted without material")

    # Forced synthesis (material exists): one last CitedAnswer call.
    yield "status", {"step": "Tool budget exhausted; synthesizing..."}
    messages.append(SystemMessage(content=FORCED_SYNTHESIS_NOTICE))
    forced_parts: list[str] = []
    async for event, data in _stream_cited_answer(bound, messages, model):
        if event == "message":
            forced_parts.append(data["text"])
        yield event, data
    yield (
        "agentic_answer",
        {
            "query": query,
            "optimized_query": query,
            "context": context,
            "sources": toolbox.ranked_sources,
            "answer": "".join(forced_parts),
        },
    )


def search_budget() -> Budget:
    """The low-budget search preset (decision 8: 1–3 iterations)."""
    return Budget(
        max_iterations=max(1, settings.AGENTIC_SEARCH_MAX_ITERATIONS),
        time_budget_seconds=settings.AGENTIC_SEARCH_TIME_BUDGET_S,
        token_budget=settings.AGENTIC_SEARCH_TOKEN_BUDGET,
    )
