"""Agentic loop over the shared pipeline tools.

The model drives `web_search` / `fetch_pages` / `get_context` in any order
until it answers through the `SubmitOutcome` tool call. The loop is bounded
by a Budget (iterations, wall-clock time, estimated tokens); exhausting a
budget degrades gracefully: whatever context the tools gathered is passed
to one forced synthesis call, never silently dropped.

Every LLM interaction is a native tool call; a free-form response is a
contract violation and becomes an SSE ``error`` event, never parsed text.
"""

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ....tools.settings import settings
from ...tools.arg_stream import ToolArgStream
from ...tools.errors import ToolProtocolError
from ...tools.toolbox import AskToolbox, build_tools
from .budgets import Budget
from .contracts import SubmitOutcome

logger = logging.getLogger(__name__)

PipelineEvent = tuple[str, dict[str, Any]]

AGENT_SYSTEM = """You are an autonomous research agent for a search engine.
Goal: answer the user's request using ONLY the provided tools.

Rules:
1. Work step by step: call `web_search` to find pages, `fetch_pages` to
   read them, `get_context` to select the passages relevant to your current
   question. Call tools in whatever order the goal requires; you may repeat
   them with different arguments.
2. Ground every claim in the tool results; never invent facts.
3. When you can answer, call `SubmitOutcome` exactly once with the final
   Markdown answer, citing sources inline as [Source](URL).
4. Every response MUST contain at least one tool call. Prose-only
   responses are a protocol violation."""

FORCED_SYNTHESIS_SYSTEM = """You are finishing an autonomous research task
whose tool budget ran out. Using STRICTLY the context chunks gathered so
far, produce the best possible final answer. Cite sources inline as
[Source](URL) and state explicitly what could not be established.
You MUST respond by calling the SubmitOutcome tool."""

BUDGET_NOTICE = (
    "The research agent stopped early ({reason}); no usable context was "
    "gathered, so no answer can be given."
)


async def _run_tool(toolbox: AskToolbox, name: str, args: dict[str, Any]) -> str:
    """Execute one bound tool by name; unknown tools return an error string."""
    tool_map = {tool.name: tool for tool in build_tools(toolbox)}
    tool = tool_map.get(name)
    if tool is None:
        return json.dumps({"error": f"Unknown tool '{name}'."})
    return await tool.coroutine(**args)


async def _gathered_context(toolbox: AskToolbox) -> str:
    """Context gathered so far: selected chunks, else all chunks."""
    chunks = toolbox.selected_chunks or toolbox.chunks
    return "\n\n---\n\n".join(chunk.get("content", "") for chunk in chunks)


async def agent_logic(
    goal: str,
    model: BaseChatModel,
    toolbox: AskToolbox,
    budget: Budget,
) -> AsyncGenerator[PipelineEvent]:
    """Run the agent loop, yielding (event, payload) pairs.

    Errors are reported as ``error`` events, never raised, so streaming
    consumers always get a well-formed termination.
    """
    try:
        tools = build_tools(toolbox)
        bound = model.bind_tools([*tools, SubmitOutcome])
        messages: list = [
            SystemMessage(content=AGENT_SYSTEM),
            HumanMessage(content=goal),
        ]

        while not budget.exhausted and budget.iterations_left:
            response = await bound.ainvoke(messages)
            budget.spend_tokens(
                response.content, json.dumps(response.tool_calls, default=str)
            )
            tool_calls = response.tool_calls or []

            outcome = next(
                (c for c in tool_calls if c.get("name") == "SubmitOutcome"), None
            )
            if outcome is not None:
                answer = str(outcome.get("args", {}).get("answer", "")).strip()
                if not answer:
                    yield (
                        "error",
                        {"error": "SubmitOutcome arrived with an empty answer."},
                    )
                    return
                yield "message", {"text": answer}
                yield "done", {"status": "completed"}
                return

            executable = [c for c in tool_calls if c.get("name") != "SubmitOutcome"]
            if not executable:
                yield (
                    "error",
                    {
                        "error": (
                            "The agent responded without a tool call; "
                            "free-form output is not accepted."
                        )
                    },
                )
                return

            messages.append(response)
            budget.spend_iteration()

            for call in executable:
                call_name = str(call.get("name", ""))
                yield "status", {"step": f"Agent tool: {call_name}(...)"}
                result = await _run_tool(toolbox, call_name, call.get("args", {}))
                messages.append(
                    ToolMessage(content=result, tool_call_id=str(call.get("id", "")))
                )

            if toolbox.ranked_sources:
                yield "sources", {"sources": toolbox.ranked_sources}
            if settings.DEBUG:
                yield "debug", {
                    "iterations_left": budget.iterations_left,
                    "tokens_spent": budget.tokens_spent,
                }

        # <----- Budget exhausted: degrade, never crash ----->
        reason = budget.stop_reason() or "budget exhausted"
        yield "status", {"step": f"Agent stopped early: {reason}."}

        context = await _gathered_context(toolbox)
        if not context.strip():
            yield "message", {"text": BUDGET_NOTICE.format(reason=reason)}
            yield "done", {"status": "budget_exhausted", "reason": reason}
            return

        try:
            async for delta in _stream_forced_outcome(model, goal, toolbox, context):
                yield "message", {"text": delta}
        except ToolProtocolError as e:
            logger.exception("Forced agent synthesis failed for goal: %s", goal)
            yield "error", {"error": str(e)}
            return
        yield "done", {"status": "budget_exhausted", "reason": reason}

    except Exception as e:
        logger.exception("Agent loop failed for goal: %s", goal)
        yield "error", {"error": f"Critical error: {str(e)}"}


async def _stream_forced_outcome(
    model: BaseChatModel, goal: str, toolbox: AskToolbox, context: str
) -> AsyncGenerator[str]:
    """One forced synthesis call over the gathered context (tool call only)."""
    sources = json.dumps(toolbox.ranked_sources)

    bound = model.bind_tools([SubmitOutcome])
    extractor = ToolArgStream(key="answer")

    async for chunk in bound.astream(
        [
            SystemMessage(
                content=(
                    f"{FORCED_SYNTHESIS_SYSTEM}\n\nSources consulted: {sources}\n"
                    f"Context chunks:\n{context}"
                )
            ),
            HumanMessage(content=goal),
        ]
    ):
        for tool_chunk in chunk.tool_call_chunks or []:
            name = tool_chunk.get("name") or ""
            if name and name != "SubmitOutcome":
                continue
            delta = extractor.feed(tool_chunk.get("args") or "")
            if delta:
                yield delta

    tail = extractor.finish()
    if tail:
        yield tail

    if not extractor.saw_value:
        raise ToolProtocolError(
            "The model did not stream the outcome through the SubmitOutcome tool."
        )
