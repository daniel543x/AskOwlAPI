"""Running-summary compression for the research loop (WU-8 / AR-3).

Merges each sub-question's fresh context into a bounded running brief, so
synthesis receives `(sub-questions + rounds) x SUBQUESTION_BRIEF_TOKENS`
instead of the raw ~25k-char concatenation that made Ollama silently
truncate and drop tool calls (settings.py documents that failure mode).
Tool call only; on failure the brief degrades to the raw concatenation —
never free-form parsing.
"""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ...tools.errors import ToolProtocolError
from .budgets import estimate_tokens
from .contracts import ContextBrief

logger = logging.getLogger(__name__)

_BRIEF_SYSTEM = """You are a research note-taker. You MUST respond by \
calling the ContextBrief tool with the updated running summary. Do not \
answer the question and do not write prose outside the tool call."""

_FIRST_PROMPT = """Create a summary of the context answering the \
sub-question: "{subquestion}".
Keep every numeric value exact (never round or generalize numbers); keep \
source URLs inline.

Context:
{context}"""

_UPDATE_PROMPT = """Update the existing running brief for the sub-question \
"{subquestion}" with the new context: integrate related information, add \
new paragraphs for new facts, ignore irrelevant material, keep every \
numeric value exact, keep source URLs inline.

Existing brief:
{brief}

New context:
{context}"""


def _cap(text: str, token_budget: int) -> str:
    """Truncate to roughly the token budget (chars/4, same as Budget)."""
    if estimate_tokens(text) <= token_budget:
        return text
    return text[: token_budget * 4]


async def compress(
    model: BaseChatModel,
    subquestion: str,
    running_brief: str,
    new_context: str,
    token_budget: int,
) -> str:
    """Merge `new_context` into the running brief for one sub-question.

    The new context input is capped at `token_budget` before the call. On
    a tool-protocol failure the brief degrades to the plain concatenation
    of the previous brief and the raw context (deterministic, no parsing).
    """
    capped_context = _cap(new_context, token_budget)
    prompt = (
        _UPDATE_PROMPT.format(
            subquestion=subquestion, brief=running_brief, context=capped_context
        )
        if running_brief.strip()
        else _FIRST_PROMPT.format(subquestion=subquestion, context=capped_context)
    )
    try:
        bound = model.bind_tools([ContextBrief])
        response = await bound.ainvoke(
            [
                SystemMessage(content=_BRIEF_SYSTEM),
                HumanMessage(content=prompt),
            ]
        )
    except ToolProtocolError:
        logger.warning(
            "Compression did not use the ContextBrief tool; "
            "degrading to raw concatenation."
        )
        return f"{running_brief}\n\n{new_context}".strip()

    for tool_call in response.tool_calls or []:
        if tool_call.get("name") == "ContextBrief":
            brief = str(tool_call.get("args", {}).get("brief", "")).strip()
            if brief:
                return brief

    logger.warning(
        "Compression returned no brief; degrading to raw concatenation."
    )
    return f"{running_brief}\n\n{new_context}".strip()
