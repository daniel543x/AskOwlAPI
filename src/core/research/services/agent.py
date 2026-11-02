"""Agentic loop over the shared pipeline tools.

The model drives `web_search` / `fetch_pages` / `get_context` in any order
until it answers through the `SubmitOutcome` tool call. The loop is bounded
by a Budget (iterations, wall-clock time, estimated tokens); exhausting a
budget degrades gracefully: whatever context the tools gathered is passed
to one forced synthesis call, never silently dropped.

Every LLM interaction is a native tool call. A free-form response is a
contract miss: the run gets one nudge to re-answer through SubmitOutcome,
and a second free-form response is salvaged as the final answer instead of
losing the run.
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import aclosing
from math import floor
from time import monotonic
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable

from ....tools.settings import settings
from ...tools.arg_stream import ToolArgStream
from ...tools.citations import CitationRewriter
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

# Sent once per run when the model answers in prose instead of tool calls.
NUDGE = "You MUST respond now by calling the SubmitOutcome tool."

# Wall-clock interval between status heartbeats while one LLM call is pending.
HEARTBEAT_SECONDS = 15.0


def _text_of(content: Any) -> str:
    """PlainText of a message content: joins text blocks, else stringifies."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)

PERSONA_TEMPLATE = """

--- Active user persona: "{name}" ---
User context (bias your search queries toward it): {system_context}
"""
PERSONA_KEYWORDS = "- Bias queries toward these keywords: {keywords}."
PERSONA_DOMAINS = (
    "- Search results are already restricted to these domains: {domains}; "
    "prefer fetching and citing pages from them."
)


def persona_system_block(persona) -> str:
    """Render the persona's biasing instructions for the system prompt."""
    block = PERSONA_TEMPLATE.format(
        name=persona.name, system_context=persona.system_context or "(none)"
    )
    if persona.allowed_keywords:
        block += "\n" + PERSONA_KEYWORDS.format(
            keywords=", ".join(persona.allowed_keywords)
        )
    if persona.allowed_domains:
        block += "\n" + PERSONA_DOMAINS.format(
            domains=", ".join(persona.allowed_domains)
        )
    return block


async def _beaten(
    model: Runnable[LanguageModelInput, AIMessage],
    messages: list[BaseMessage],
) -> AsyncGenerator[PipelineEvent]:
    """One `ainvoke` round, yielding a status heartbeat while it pends.

    Ticks stream out as they occur; the pending task is cancelled if the
    consumer abandons the generator mid-call.
    """
    started = monotonic()
    task: asyncio.Task[AIMessage] = asyncio.create_task(model.ainvoke(messages))
    try:
        while True:
            try:
                response = await asyncio.wait_for(
                    asyncio.shield(task), timeout=HEARTBEAT_SECONDS
                )
                yield "__llm_done__", {"response": response}
                return
            except TimeoutError:
                elapsed = floor(monotonic() - started)
                yield "status", {"step": f"Agent thinking… ({elapsed}s)"}
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except BaseException:  # noqa: BLE001 - task cleanup, any outcome
                pass


async def _run_tool(toolbox: AskToolbox, name: str, args: dict[str, Any]) -> str:
    """Execute one bound tool by name; unknown tools return an error string."""
    tool_map = {tool.name: tool for tool in build_tools(toolbox)}
    tool = tool_map.get(name)
    if tool is None:
        return json.dumps({"error": f"Unknown tool '{name}'."})
    if tool.coroutine is None:
        return json.dumps({"error": f"Tool '{name}' is not async-callable."})
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
    persona=None,
    images: list[bytes] | None = None,
) -> AsyncGenerator[PipelineEvent]:
    """Run the agent loop, yielding (event, payload) pairs.

    When a persona is passed, its system context biases the agent's query
    rewriting (system prompt) while its domain/engine constraints scope
    the toolbox.

    Errors are reported as ``error`` events, never raised, so streaming
    consumers always get a well-formed termination.
    """
    try:
        tools = build_tools(toolbox)
        bound = model.bind_tools([*tools, SubmitOutcome])
        system = AGENT_SYSTEM
        if persona is not None:
            system += persona_system_block(persona)
        messages: list[BaseMessage] = [
            SystemMessage(content=system),
            HumanMessage(content=goal),
        ]

        nudge_sent = False
        while not budget.exhausted and budget.iterations_left:
            response: AIMessage | None = None
            async with aclosing(_beaten(bound, messages)) as beat:
                async for tick in beat:
                    if tick[0] == "status":
                        yield tick
                    else:
                        response = tick[1]["response"]
            if response is None:
                raise ToolProtocolError(
                    "The agent LLM call ended without a response."
                )
            budget.spend_tokens(
                str(response.content), json.dumps(response.tool_calls, default=str)
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
                known = [s["url"] for s in toolbox.ranked_sources]
                rewriter = CitationRewriter(known)
                answer = rewriter.feed(answer) + rewriter.flush()
                yield "message", {"text": answer}
                yield "done", {"status": "completed"}
                return

            executable = [c for c in tool_calls if c.get("name") != "SubmitOutcome"]
            if not executable:
                text = _text_of(response.content).strip()
                messages.append(response)
                if text and not nudge_sent:
                    nudge_sent = True
                    messages.append(HumanMessage(content=NUDGE))
                    budget.spend_iteration()
                    continue
                if text:
                    yield "message", {"text": text}
                    yield "done", {"status": "completed"}
                    return
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
            new_offers = toolbox.take_new_offers()
            if new_offers:
                yield "offers", {"offers": new_offers}
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
            async for delta in _stream_forced_outcome(
            model, goal, toolbox, context, images
        ):
                yield "message", {"text": delta}
        except ToolProtocolError as e:
            logger.exception("Forced agent synthesis failed for goal: %s", goal)
            yield "error", {"error": str(e)}
            return
        yield "done", {"status": "budget_exhausted", "reason": reason}

    except Exception as e:
        logger.exception("Agent loop failed for goal: %s", goal)
        yield "error", {"error": f"Critical error: {str(e)}"}


def image_blocks_for(images: list[bytes] | None) -> list[dict]:
    """LiteLLM/OpenAI image content blocks (PR 16)."""
    import base64

    return [
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,"
                + base64.b64encode(data).decode("ascii")
            },
        }
        for data in images or []
    ]


async def _stream_forced_outcome(
    model: BaseChatModel,
    goal: str,
    toolbox: AskToolbox,
    context: str,
    images: list[bytes] | None = None,
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
            HumanMessage(
                content=(
                    [{"type": "text", "text": goal}, *image_blocks_for(images)]
                    if images
                    else goal
                )
            ),
        ]
    ):
        if not isinstance(chunk, AIMessageChunk):
            continue
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
