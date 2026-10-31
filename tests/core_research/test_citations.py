"""Offline tests: numbered citations in research report assembly."""

import uuid

from src.core.research.services.contracts import ReportSection, SubmitReport
from src.core.research.services.synthesis import (
    assemble_note_report,
    assemble_note_report_stream,
)
from src.core.tools.citations import (
    body_before_sources,
    validate_citations,
)
from src.modules.notes.model import ResearchNote

UA = "https://example.com/a"
UB = "https://example.com/b"
UC = "https://example.com/c"


def _note(sequence: int, task: str, content: str, sources: list[str]):
    return ResearchNote(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        sequence=sequence,
        task=task,
        content=content,
        sources_used=list(sources),
    )


def _plan(*idxs: int, intro: str = "Intro.") -> SubmitReport:
    return SubmitReport(
        intro=intro,
        sections=[
            ReportSection(idx=i, order=position)
            for position, i in enumerate(idxs, 1)
        ],
    )


async def _collect(notes, plan) -> str:
    """The stream is an async generator (the pipeline does `async for`)."""
    return "".join([chunk async for chunk in assemble_note_report_stream(notes, plan)])


async def test_assembly_numbers_by_first_appearance_and_lists_sources():
    notes = [
        _note(1, "T1", f"Owns [Source]({UB}) then [Source]({UA}).", [UA, UB]),
        _note(2, "T2", f"Also [Source]({UC}).", [UC]),
    ]
    plan = _plan(1, 2)

    report = await _collect(notes, plan)
    assert assemble_note_report(notes, plan) == report

    assert "[1]" in report and "[2]" in report and "[3]" in report
    assert "[Source](" not in report
    assert "Citations: [2] [1]" in report  # note 1's sources_used order
    assert "Citations: [3]" in report
    assert f"- [1] {UB}" in report
    assert f"- [2] {UA}" in report
    assert f"- [3] {UC}" in report
    assert report.index("## Sources") < report.index("- [1]")


async def test_assembly_appends_consulted_but_uncited():
    # Beyond the 8-URL section display cap, the tail is still listed and
    # numbered (never a hole), just never cited in prose.
    many = [f"https://example.com/s{i}" for i in range(1, 10)]
    notes = [_note(1, "T1", f"Only [Source]({many[0]}) cited.", many)]
    report = await _collect(notes, _plan(1))

    assert "[1]" in report
    assert f"- [9] {many[8]}" in report  # consulted, beyond the cap

    body = body_before_sources(report)
    registry = {url: i + 1 for i, url in enumerate(many)}
    verdict = validate_citations(body, registry)
    assert verdict.uncited_sources == [9]  # beyond the display cap
    assert verdict.missing_numbers == []


async def test_validator_scans_body_not_sources_section():
    many = [f"https://example.com/s{i}" for i in range(1, 10)]
    notes = [_note(1, "T1", "Prose cites [1] directly.", many)]
    report = await _collect(notes, _plan(1))

    body = body_before_sources(report)
    registry = {url: i + 1 for i, url in enumerate(many)}
    verdict = validate_citations(body, registry)
    assert verdict.uncited_sources == [9]


async def test_no_sources_report_is_unchanged():
    notes = [_note(1, "T1", "Plain body, no sources.", [])]
    plan = _plan(1, intro="Intro.")
    report = await _collect(notes, plan)

    assert report == "Intro.\n\n## T1\n\nPlain body, no sources.\n\n"
    assert "## Sources" not in report


async def test_stream_and_sync_assembly_match():
    notes = [
        _note(1, "T1", f"Body [Source]({UA}).", [UA, UB]),
        _note(2, "T2", "No citations here.", [UB]),
    ]
    plan = _plan(2, 1, intro="Reversed order.")
    assert assemble_note_report(notes, plan) == await _collect(notes, plan)


async def test_fallback_report_rewrites_against_gathered_sources():
    from src.core.research.services.synthesis import generate_report
    from tests.core_ask.fakes import ScriptedToolModel, tool_call

    gathered = [
        {
            "subquestion": "rust safety",
            "sources": [{"url": UA, "title": 'Rust page, "quoted"'}],
            "context": "Body citing [Source](https://example.com/a).",
        }
    ]
    model = ScriptedToolModel(
        responses=[tool_call("ResearchReport", {"report": f"Cites [Source]({UA})."})]
    )
    report = await generate_report(model, "rust", gathered)
    assert report == "Cites [1]."
