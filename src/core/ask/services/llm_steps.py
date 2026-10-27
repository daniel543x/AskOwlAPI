"""LLM steps of the ask pipeline, all via native tool calls.

Rule enforced here: the LLM is never trusted with free-form output.
- Query rewriting must arrive as a `SubmitSearchQuery` tool call; the only
  fallback is the raw user query (a tool-protocol error path).
- Answer synthesis must arrive as a `CitedAnswer` tool call; anything else is
  a contract violation and an error.
"""

import logging
from collections.abc import AsyncGenerator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)

from ....tools.dates import current_date_line
from ...tools.arg_stream import ToolArgStream
from ...tools.errors import ToolProtocolError
from .contracts import CitedAnswer, SubmitSearchQuery

logger = logging.getLogger(__name__)

__all__ = [
    "ToolProtocolError",
    "rewrite_query",
    "generate_cited_answer",
    "stream_cited_answer",
]


QUERY_OPTIMIZER_SYSTEM = """You are a Web Search Query Optimizer.
Convert the user's question into an optimized search query:
1. Noise Removal: remove conversational filler.
2. Keyword Optimization: focus on entities and nouns.
3. Language Preservation: keep the user's language.
4. Standalone Resolution: when a conversation is provided, resolve pronouns
   and omissions using it, so the query works as a standalone web search.
You MUST respond by calling the SubmitSearchQuery tool with the optimized
query. Do not answer the question and do not write any prose."""

ANSWER_SYSTEM_TEMPLATE = """You are an expert AI assistant for a search engine.
Answer the user's question using STRICTLY the context chunks below, which
were gathered by the search tools. Each chunk is prefixed with its citation
number, e.g. `[1]`.

Guidelines:
1. Base the answer ONLY on the provided context.
2. Cite claims with bracketed numbers referring to those context chunks,
   e.g. `[1]` or `[3]`, placed right after each claim. Cite only numbers
   that exist among the context chunks; every factual claim needs at least
   one citation.
3. If the context does not contain the answer, say so explicitly.
4. You MUST respond by calling the CitedAnswer tool with the final answer.

{current_date}

Context Chunks:
{context}"""


# <----- Query rewriting ----->


def _history_block(history: list[tuple[str, str]]) -> str:
    """The last conversation turn pair, truncated, for the rewriter."""
    last_user = ""
    last_assistant = ""
    for role, content in history:
        if role == "user":
            last_user = content
        elif role == "assistant":
            last_assistant = content
    return (
        "Conversation so far:\n"
        f"User: {last_user[:1000]}\n"
        f"Assistant: {last_assistant[:500]}\n\n"
    )


async def rewrite_query(
    model: BaseChatModel,
    raw_query: str,
    history: list[tuple[str, str]] | None = None,
    today: str | None = None,
) -> tuple[str, bool, bool]:
    """Optimize the query via a native tool call (WU-1 / AR-4).

    With `history`, the last turn pair is prepended so pronouns and
    omissions resolve into a standalone query. Returns
    (optimized_query, used_tool_call, standalone): on a tool-protocol error
    the raw query is used (standalone False) — free-form model text is
    never parsed or trusted.
    """
    bound = model.bind_tools([SubmitSearchQuery])
    prompt = raw_query
    if history:
        prompt = f"{_history_block(history)}Current question: {raw_query}"
    response = await bound.ainvoke(
        [
            SystemMessage(
                content=f"{current_date_line(today)}\n{QUERY_OPTIMIZER_SYSTEM}"
            ),
            HumanMessage(content=prompt),
        ]
    )

    for tool_call in response.tool_calls or []:
        if tool_call.get("name") == "SubmitSearchQuery":
            args = tool_call.get("args", {})
            optimized = str(args.get("query", "")).strip()
            if optimized:
                standalone = bool(args.get("standalone", True))
                return optimized, True, standalone

    logger.warning(
        "Query rewriting did not use the SubmitSearchQuery tool; "
        "falling back to the raw query (tool-protocol error path)."
    )
    return raw_query, False, False


# <----- Answer synthesis ----->


def _answer_messages(
    query: str,
    context: str,
    history: list[tuple[str, str]] | None = None,
    today: str | None = None,
) -> list:
    """System (with current date) + prior turns + question."""
    messages: list = [
        SystemMessage(
            content=ANSWER_SYSTEM_TEMPLATE.format(
                context=context, current_date=current_date_line(today)
            )
        )
    ]
    for role, content in history or []:
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=query))
    return messages


async def generate_cited_answer(
    model: BaseChatModel,
    query: str,
    context: str,
    history: list[tuple[str, str]] | None = None,
    today: str | None = None,
) -> str:
    """Non-streaming cited answer via the CitedAnswer tool call."""
    bound = model.bind_tools([CitedAnswer])
    response = await bound.ainvoke(_answer_messages(query, context, history, today))

    for tool_call in response.tool_calls or []:
        if tool_call.get("name") == "CitedAnswer":
            answer = str(tool_call.get("args", {}).get("answer", "")).strip()
            if answer:
                return answer

    raise ToolProtocolError(
        "The model did not return the answer through the CitedAnswer tool."
    )


async def stream_cited_answer(
    model: BaseChatModel,
    query: str,
    context: str,
    history: list[tuple[str, str]] | None = None,
    today: str | None = None,
) -> AsyncGenerator[str]:
    """Stream the `answer` argument of the CitedAnswer tool call.

    Providers stream tool-call arguments as JSON fragments; ToolArgStream
    decodes the target string incrementally so SSE consumers get text deltas.
    """
    bound = model.bind_tools([CitedAnswer])
    extractor = ToolArgStream(key="answer")

    async for chunk in bound.astream(
        _answer_messages(query, context, history, today)
    ):
        if not isinstance(chunk, AIMessageChunk):
            continue
        for tool_chunk in chunk.tool_call_chunks or []:
            name = tool_chunk.get("name") or ""
            if name and name != "CitedAnswer":
                continue
            delta = extractor.feed(tool_chunk.get("args") or "")
            if delta:
                yield delta

    tail = extractor.finish()
    if tail:
        yield tail

    if not extractor.saw_value:
        raise ToolProtocolError(
            "The model did not stream the answer through the CitedAnswer tool."
        )
