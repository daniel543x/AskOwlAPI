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
from langchain_core.messages import HumanMessage, SystemMessage

from .contracts import CitedAnswer, SubmitSearchQuery

logger = logging.getLogger(__name__)


class ToolProtocolError(Exception):
    """The model did not answer through the required tool call."""


QUERY_OPTIMIZER_SYSTEM = """You are a Web Search Query Optimizer.
Convert the user's question into an optimized search query:
1. Noise Removal: remove conversational filler.
2. Keyword Optimization: focus on entities and nouns.
3. Language Preservation: keep the user's language.
You MUST respond by calling the SubmitSearchQuery tool with the optimized
query. Do not answer the question and do not write any prose."""

ANSWER_SYSTEM_TEMPLATE = """You are an expert AI assistant for a search engine.
Answer the user's question using STRICTLY the context chunks below, which
were gathered by the search tools.

Guidelines:
1. Base the answer ONLY on the provided context.
2. Cite sources with Markdown links: [Source](URL) right after each claim.
3. If the context does not contain the answer, say so explicitly.
4. You MUST respond by calling the CitedAnswer tool with the final answer.

Context Chunks:
{context}"""


# <----- Query rewriting ----->


async def rewrite_query(model: BaseChatModel, raw_query: str) -> tuple[str, bool]:
    """Optimize the query via a native tool call.

    Returns (optimized_query, used_tool_call). On a tool-protocol error the
    raw query is used; free-form model text is never parsed or trusted.
    """
    bound = model.bind_tools([SubmitSearchQuery])
    response = await bound.ainvoke(
        [
            SystemMessage(content=QUERY_OPTIMIZER_SYSTEM),
            HumanMessage(content=raw_query),
        ]
    )

    for tool_call in response.tool_calls or []:
        if tool_call.get("name") == "SubmitSearchQuery":
            optimized = str(tool_call.get("args", {}).get("query", "")).strip()
            if optimized:
                return optimized, True

    logger.warning(
        "Query rewriting did not use the SubmitSearchQuery tool; "
        "falling back to the raw query (tool-protocol error path)."
    )
    return raw_query, False


# <----- Answer synthesis ----->


def _answer_messages(query: str, context: str) -> list:
    return [
        SystemMessage(content=ANSWER_SYSTEM_TEMPLATE.format(context=context)),
        HumanMessage(content=query),
    ]


async def generate_cited_answer(
    model: BaseChatModel, query: str, context: str
) -> str:
    """Non-streaming cited answer via the CitedAnswer tool call."""
    bound = model.bind_tools([CitedAnswer])
    response = await bound.ainvoke(_answer_messages(query, context))

    for tool_call in response.tool_calls or []:
        if tool_call.get("name") == "CitedAnswer":
            answer = str(tool_call.get("args", {}).get("answer", "")).strip()
            if answer:
                return answer

    raise ToolProtocolError(
        "The model did not return the answer through the CitedAnswer tool."
    )


async def stream_cited_answer(
    model: BaseChatModel, query: str, context: str
) -> AsyncGenerator[str]:
    """Stream the `answer` argument of the CitedAnswer tool call.

    Providers stream tool-call arguments as JSON fragments; ToolArgStream
    decodes the target string incrementally so SSE consumers get text deltas.
    """
    bound = model.bind_tools([CitedAnswer])
    extractor = ToolArgStream(key="answer")

    async for chunk in bound.astream(_answer_messages(query, context)):
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


# <----- Streaming tool-argument extraction ----->

_JSON_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "b": "\b",
    "f": "\f",
}


class ToolArgStream:
    """Incrementally decodes one string value from streamed tool-call args.

    Providers stream tool-call arguments as fragments of a JSON object, e.g.
    `{"an`, `swer": "Rust`, ` is..."}`. This state machine tolerates
    arbitrary fragment boundaries (including inside escape sequences and
    unicode escapes) and emits the decoded value incrementally.
    """

    def __init__(self, key: str):
        self._tokens = ('"', key, '"', ':', '"')
        self._token_index = 0
        self._in_token = False
        self._token_pos = 0
        self._state = "match"  # match | value | escape | unicode | done
        self._unicode_digits = ""
        self._decoded: list[str] = []
        self._emitted = 0
        self._pending = ""
        self.saw_value = False

    def feed(self, fragment: str) -> str:
        """Consume a fragment; return newly decoded text."""
        self._pending += fragment
        self._process()
        return self._emit()

    def finish(self) -> str:
        """Mark the stream complete; flush any decoded-but-unemitted text."""
        if self._state in ("value", "escape", "unicode"):
            self._state = "done"
        return self._emit()

    # <----- internals ----->

    def _emit(self) -> str:
        new_text = "".join(self._decoded[self._emitted :])
        self._emitted = len(self._decoded)
        return new_text

    def _process(self) -> None:
        while self._pending and self._state != "done":
            char = self._pending[0]
            self._pending = self._pending[1:]

            if self._state == "match":
                self._match(char)
            elif self._state == "value":
                self._value(char)
            elif self._state == "escape":
                self._escape(char)
            elif self._state == "unicode":
                self._unicode(char)

    def _match(self, char: str) -> None:
        token = self._tokens[self._token_index]

        if not self._in_token:
            # Whitespace between tokens is skipped.
            if char.isspace():
                return
            if char != token[0]:
                # Not the expected token start; restart matching.
                self._token_index = 0
                return
            self._in_token = True
            self._token_pos = 1
            if self._token_pos == len(token):
                self._advance_token()
            return

        # Inside a token: characters must match exactly.
        if self._token_pos < len(token) and char == token[self._token_pos]:
            self._token_pos += 1
            if self._token_pos == len(token):
                self._advance_token()
        else:
            # Mismatch: restart from the first token.
            self._token_index = 0
            self._in_token = False

    def _advance_token(self) -> None:
        self._in_token = False
        if self._token_index == len(self._tokens) - 1:
            # Consumed the opening quote of the value: value starts here.
            self._state = "value"
            self.saw_value = True
        else:
            self._token_index += 1

    def _value(self, char: str) -> None:
        if char == "\\":
            self._state = "escape"
        elif char == '"':
            self._state = "done"
        else:
            self._decoded.append(char)

    def _escape(self, char: str) -> None:
        if char == "u":
            self._state = "unicode"
            self._unicode_digits = ""
        else:
            self._decoded.append(_JSON_ESCAPES.get(char, char))
            self._state = "value"

    def _unicode(self, char: str) -> None:
        self._unicode_digits += char
        if len(self._unicode_digits) == 4:
            try:
                self._decoded.append(chr(int(self._unicode_digits, 16)))
            except ValueError:
                self._decoded.append(self._unicode_digits)
            self._state = "value"
