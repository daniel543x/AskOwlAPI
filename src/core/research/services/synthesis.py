"""Research synthesis: a grounded, cited report via a native tool call."""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ...tools.arg_stream import ToolArgStream
from ...tools.errors import ToolProtocolError
from .contracts import ResearchReport

logger = logging.getLogger(__name__)

REPORT_SYSTEM_TEMPLATE = """You are a research assistant compiling a report
for a search engine. You receive sub-questions, the context chunks gathered
for each, and the sources that were consulted.

Guidelines:
1. Write one Markdown section (## heading = the sub-question) per
   sub-question that has context.
2. Ground every claim STRICTLY in the provided context chunks and cite
   sources inline as [Source](URL) right after each claim.
3. Do not invent facts. If the context does not answer a sub-question, say
   so explicitly in that section.
4. End with a short "## Summary" of what was established.
5. You MUST respond by calling the ResearchReport tool with the report.

Context per sub-question:
{context}"""

NO_CONTEXT_REPORT = (
    "No relevant content could be extracted from the search results for "
    "any sub-question, so no research report can be given."
)


def _format_gathered(gathered: list[dict[str, Any]]) -> str:
    sections = []
    for index, item in enumerate(gathered, start=1):
        sources = ", ".join(
            source.get("url", "") for source in item.get("sources", [])
        )
        sections.append(
            f"### Sub-question {index}: {item['subquestion']}\n"
            f"Sources consulted: {sources or 'none'}\n"
            f"Context chunks:\n{item['context']}"
        )
    return "\n\n".join(sections)


def _report_messages(query: str, gathered: list[dict[str, Any]]) -> list:
    return [
        SystemMessage(
            content=REPORT_SYSTEM_TEMPLATE.format(
                context=_format_gathered(gathered)
            )
        ),
        HumanMessage(content=query),
    ]


async def generate_report(
    model: BaseChatModel, query: str, gathered: list[dict[str, Any]]
) -> str:
    """Non-streaming research report via the ResearchReport tool call."""
    bound = model.bind_tools([ResearchReport])
    response = await bound.ainvoke(_report_messages(query, gathered))

    for tool_call in response.tool_calls or []:
        if tool_call.get("name") == "ResearchReport":
            report = str(tool_call.get("args", {}).get("report", "")).strip()
            if report:
                return report

    raise ToolProtocolError(
        "The model did not return the report through the ResearchReport tool."
    )


async def stream_report(
    model: BaseChatModel, query: str, gathered: list[dict[str, Any]]
) -> AsyncGenerator[str]:
    """Stream the `report` argument of the ResearchReport tool call."""
    bound = model.bind_tools([ResearchReport])
    extractor = ToolArgStream(key="report")

    async for chunk in bound.astream(_report_messages(query, gathered)):
        for tool_chunk in chunk.tool_call_chunks or []:
            name = tool_chunk.get("name") or ""
            if name and name != "ResearchReport":
                continue
            delta = extractor.feed(tool_chunk.get("args") or "")
            if delta:
                yield delta

    tail = extractor.finish()
    if tail:
        yield tail

    if not extractor.saw_value:
        raise ToolProtocolError(
            "The model did not stream the report through the ResearchReport tool."
        )
