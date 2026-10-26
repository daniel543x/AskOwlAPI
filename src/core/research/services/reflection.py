"""Reflection step of the research loop (WU-7 / AR-3).

Reviews the gathered context after a full sub-question pass and decides
whether anything is still missing. Tool call only; a tool-protocol error
returns an empty `MissingInfo` (deterministic stop — never parse prose).
"""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ...tools.errors import ToolProtocolError
from .contracts import MissingInfo

logger = logging.getLogger(__name__)

REFLECTION_SYSTEM = """You are reviewing a research progress report: the \
question, followed by one gathered context section per sub-question.
Identify concrete, missing information that the report needs to answer the \
question. If the gathered context already answers it, return empty lists.
You MUST respond by calling the MissingInfo tool. Do not answer the \
question and do not write prose."""


async def reflect(
    model: BaseChatModel,
    query: str,
    gathered: list[dict],
) -> MissingInfo:
    """Decide what is missing after a gathering pass.

    `gathered` carries `{subquestion, context}` entries (briefs when
    compression is on). On any tool-protocol failure the result is an empty
    `MissingInfo`, so the loop stops deterministically instead of guessing.
    """
    bound = model.bind_tools([MissingInfo])
    sections = "\n\n".join(
        f"## Sub-question: {entry.get('subquestion', '')}\n"
        f"{entry.get('context', '')}"
        for entry in gathered
    )
    try:
        response = await bound.ainvoke(
            [
                SystemMessage(content=REFLECTION_SYSTEM),
                HumanMessage(
                    content=f"Research question: {query}\n\n"
                    f"Progress report:\n{sections}"
                ),
            ]
        )
    except ToolProtocolError:
        logger.warning(
            "Reflection did not use the MissingInfo tool; stopping the loop."
        )
        return MissingInfo(missing=[], follow_up_queries=[])

    for tool_call in response.tool_calls or []:
        if tool_call.get("name") == "MissingInfo":
            args = tool_call.get("args", {})
            return MissingInfo(
                missing=[str(m) for m in args.get("missing", []) if str(m).strip()],
                follow_up_queries=[
                    str(q)
                    for q in args.get("follow_up_queries", [])
                    if str(q).strip()
                ][:3],
            )

    logger.warning(
        "Reflection returned no MissingInfo tool call; stopping the loop."
    )
    return MissingInfo(missing=[], follow_up_queries=[])
