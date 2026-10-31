"""Research synthesis: a grounded, cited report via a native tool call."""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from ....tools.dates import current_date_line
from ...tools.arg_stream import ToolArgStream
from ...tools.citations import (
    MARKDOWN_LINK,
    CitationRewriter,
    build_registry,
    rewrite_citations,
    validate_citations,
)
from ...tools.errors import ToolProtocolError
from .contracts import ResearchReport, SubmitReport

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


def _image_blocks(images: list[bytes] | None) -> list[dict]:
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


def _report_messages(
    query: str,
    gathered: list[dict[str, Any]],
    today: str | None = None,
    images: list[bytes] | None = None,
) -> list[BaseMessage]:
    image_blocks = _image_blocks(images)
    question: str | list = query
    if image_blocks:
        question = [{"type": "text", "text": query}, *image_blocks]
    return [
        SystemMessage(
            content=f"{current_date_line(today)}\n"
            + REPORT_SYSTEM_TEMPLATE.format(context=_format_gathered(gathered))
        ),
        HumanMessage(content=question),
    ]


def _urls_in(gathered: list[dict[str, Any]]) -> list[str]:
    """The gathered sources' URLs, in order of appearance.

    Explicit traversal: scraping URLs out of the serialized blob would
    drag in JSON punctuation and prose fragments as fake sources.
    """
    seen: list[str] = []
    for item in gathered or []:
        for source in item.get("sources") or []:
            url = (source or {}).get("url") if isinstance(source, dict) else None
            if url and url not in seen:
                seen.append(url)
    return seen


async def generate_report(
    model: BaseChatModel,
    query: str,
    gathered: list[dict[str, Any]],
    today: str | None = None,
    images: list[bytes] | None = None,
) -> str:
    """Non-streaming research report via the ResearchReport tool call."""
    bound = model.bind_tools([ResearchReport])
    response = await bound.ainvoke(
        _report_messages(query, gathered, today, images)
    )

    for tool_call in response.tool_calls or []:
        if tool_call.get("name") == "ResearchReport":
            report = str(tool_call.get("args", {}).get("report", "")).strip()
            if report:
                rewriter = CitationRewriter(_urls_in(gathered))
                return rewriter.feed(report) + rewriter.flush()

    raise ToolProtocolError(
        "The model did not return the report through the ResearchReport tool."
    )


async def stream_report(
    model: BaseChatModel,
    query: str,
    gathered: list[dict[str, Any]],
    today: str | None = None,
    images: list[bytes] | None = None,
) -> AsyncGenerator[str]:
    """Stream the `report` argument of the ResearchReport tool call."""
    bound = model.bind_tools([ResearchReport])
    extractor = ToolArgStream(key="report")
    rewriter = CitationRewriter(_urls_in(gathered))

    async for chunk in bound.astream(
        _report_messages(query, gathered, today, images)
    ):
        if not isinstance(chunk, AIMessageChunk):
            continue
        for tool_chunk in chunk.tool_call_chunks or []:
            name = tool_chunk.get("name") or ""
            if name and name != "ResearchReport":
                continue
            delta = extractor.feed(tool_chunk.get("args") or "")
            if delta:
                text = rewriter.feed(delta)
                if text:
                    yield text

    tail = extractor.finish()
    if tail:
        text = rewriter.feed(tail) + rewriter.flush()
        if text:
            yield text

    if not extractor.saw_value:
        raise ToolProtocolError(
            "The model did not stream the report through the ResearchReport tool."
        )


# <----- Note-based report assembly (PR 18 / decision 9) ----->

EDITORIAL_SYSTEM = """You are the editor of a research report.
You receive the user's query and the TOC of the research notes written
during the run (idx, task, one-line summary, sources count) — never the
full note contents.

Call `SubmitReport` exactly once with the section plan: include the
notes that answer the query, order them into a coherent narrative
(order starts at 1), and optionally give each included section a
one-sentence lead-in. Add a short intro paragraph when useful.
You MUST respond by calling the SubmitReport tool; prose is a protocol
violation."""


def toc_block(notes: list) -> str:
    """TOC-only view for the editorial call (never the contents)."""
    from ....modules.notes.model import toc_entry

    lines = [
        "- idx {idx}: {task} — {summary} (sources: {count})".format(
            idx=entry["idx"],
            task=entry["task"],
            summary=entry["one_line_summary"],
            count=entry["sources_count"],
        )
        for entry in (toc_entry(note) for note in notes)
    ]
    return "\n".join(lines) or "(no notes were written)"


def _editorial_messages(query: str, notes: list) -> list:
    return [
        SystemMessage(content=EDITORIAL_SYSTEM),
        HumanMessage(
            content=(
                f"Query: {query}\n\nResearch notes TOC:\n{toc_block(notes)}"
            )
        ),
    ]


async def plan_note_report(model: BaseChatModel, query: str, notes: list):
    """Editorial call: the TOC-only SubmitReport plan (tool call only)."""
    bound = model.bind_tools([SubmitReport])
    response = await bound.ainvoke(_editorial_messages(query, notes))
    for tool_call in response.tool_calls or []:
        if tool_call.get("name") == "SubmitReport":
            return SubmitReport.model_validate(tool_call.get("args", {}))
    raise ToolProtocolError(
        "The editorial step did not use the SubmitReport tool."
    )


def _report_parts(notes: list, plan) -> list[str]:
    """Intro + section units, unnumbered (the assembly is deterministic)."""
    parts: list[str] = []
    if plan.intro:
        parts.append(f"{plan.intro}\n\n")
    by_idx = {note.sequence: note for note in notes}
    for section in sorted(
        (s for s in plan.sections if s.include),
        key=lambda s: s.order,
    ):
        note = by_idx.get(section.idx)
        if note is None:
            continue
        parts.append(f"## {note.task}\n\n")
        if section.lead_in:
            parts.append(f"{section.lead_in}\n\n")
        body = note.content.strip() or "(no content recorded)"
        parts.append(f"{body}\n\n")
        urls = list(note.sources_used or [])[:8]
        if urls:
            links = " ".join(f"[Source]({url})" for url in urls)
            parts.append(f"Citations: {links}\n\n")
    return parts


def _numbered_report(parts: list[str], notes: list) -> tuple[str, dict[str, int]]:
    """Rewrite the assembled parts to dense `[n]` and build `## Sources`.

    Registry: first appearance of any consulted URL in the body, then
    consulted-but-uncited URLs appended in note order — the cited prefix
    stays dense by construction.
    """
    body = "".join(parts)
    known: list[str] = []
    for note in notes:
        for url in note.sources_used or []:
            if url not in known:
                known.append(url)
    cited_order: list[str] = []
    for match in MARKDOWN_LINK.finditer(body):
        url = match.group("url")
        if url in known and url not in cited_order:
            cited_order.append(url)
    registry = build_registry(known, cited_order)
    rewritten, _ = rewrite_citations(body, registry)
    if registry:
        verdict = validate_citations(body, registry)
        logger.debug(
            "Citation verdict: cited=%s missing=%s unresolved=%s uncited=%s",
            verdict.cited_numbers,
            verdict.missing_numbers,
            verdict.unresolved_markers,
            verdict.uncited_sources,
        )
        rewritten += "## Sources\n\n" + "".join(
            f"- [{n}] {url}\n" for url, n in registry.items()
        )
    return rewritten, registry


async def assemble_note_report_stream(
    notes: list, plan
) -> AsyncGenerator[str]:
    """Deterministic assembly: numbered intro + sections, then Sources."""
    parts = _report_parts(notes, plan)
    body = "".join(parts)
    known: list[str] = []
    for note in notes:
        for url in note.sources_used or []:
            if url not in known:
                known.append(url)
    cited_order: list[str] = []
    for match in MARKDOWN_LINK.finditer(body):
        url = match.group("url")
        if url in known and url not in cited_order:
            cited_order.append(url)
    registry = build_registry(known, cited_order)
    for part in parts:
        rewritten, _ = rewrite_citations(part, registry)
        yield rewritten
    if registry:
        yield "## Sources\n\n" + "".join(
            f"- [{n}] {url}\n" for url, n in registry.items()
        )


def assemble_note_report(notes: list, plan) -> str:
    """Synchronous assembly (the JSON adapter path)."""
    report, _ = _numbered_report(_report_parts(notes, plan), notes)
    return report

