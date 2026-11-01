"""Offline tests for research notes + note-based reporting (PR 18)."""

import json
import uuid

import pytest
from langchain_core.messages import AIMessage

from src.core.research.services.contracts import SubmitReport
from src.core.research.services.pipeline import research_logic
from src.core.research.services.synthesis import (
    assemble_note_report,
    plan_note_report,
    toc_block,
)
from src.modules.notes.model import ResearchNote
from src.modules.notes.store import NoteStore
from tests.core_ask.fakes import (
    FakeRanker,
    FakeScraper,
    FakeSearchProvider,
    ScriptedToolModel,
    tool_call,
)

USER_ID = uuid.uuid4()


class FakeNoteRepo:
    """In-memory NoteRepository double; sequences per store."""

    def __init__(self):
        self.rows: list[ResearchNote] = []

    async def create(self, user_id, task, content, sources_used,
                     conversation_id=None):
        note = ResearchNote(
            id=uuid.uuid4(),
            user_id=user_id,
            conversation_id=conversation_id,
            sequence=len(self.rows) + 1,
            task=task,
            content=content,
            sources_used=list(sources_used),
        )
        self.rows.append(note)
        return note

    async def get_by_ids(self, note_ids):
        by_id = {n.id: n for n in self.rows}
        return [by_id[i] for i in note_ids if i in by_id]


def _make_toolbox(note_store=None):
    from src.core.tools.toolbox import AskToolbox

    pages = {
        "http://example.com/a": "# Rust ownership\\nOwnership makes rust safe.",
        "http://example.com/b": "# Borrow checker\\nBorrows are checked.",
    }
    return AskToolbox(
        FakeSearchProvider(), FakeScraper(pages=pages), FakeRanker(),
        None, None, note_store=note_store,
    )


# <----- NoteStore + TOC ----->


async def test_note_store_persists_and_returns_toc_entries():
    repo = FakeNoteRepo()
    store = NoteStore(repo, USER_ID)
    entry = await store.write("ownership", "Notes about ownership.", ["u1"])

    assert entry["idx"] == 1
    assert entry["task"] == "ownership"
    assert entry["sources_count"] == 1
    assert "id" in entry
    second = await store.write("borrows", "Borrow notes.", ["u2", "u3"])
    assert second["idx"] == 2
    assert [n.sequence for n in store.notes] == [1, 2]


def test_toc_block_hides_contents():
    note = ResearchNote(
        sequence=1, task="ownership", content="SECRET" * 100,
        sources_used=["u1"],
    )
    block = toc_block([note])
    # The TOC carries a one-line summary only: the full note contents
    # never reach the editorial call.
    assert "ownership" in block
    assert ("SECRET" * 100) not in block  # full contents never leak


# <----- note_write tool ----->


async def test_note_write_tool_persists_and_returns_toc():
    from src.core.tools.toolbox import build_tools

    repo = FakeNoteRepo()
    toolbox = _make_toolbox(note_store=NoteStore(repo, USER_ID))
    tool = next(t for t in build_tools(toolbox) if t.name == "note_write")

    result = await tool.coroutine(
        task="ownership", content="Grounded facts.", sources_used=["u1"]
    )
    payload = json.loads(result)
    assert payload["wrote"]["idx"] == 1
    assert payload["toc"][0]["task"] == "ownership"


def test_note_write_absent_without_store():
    from src.core.tools.toolbox import build_tools

    toolbox = _make_toolbox()
    assert all(t.name != "note_write" for t in build_tools(toolbox))


# <----- Editorial plan + deterministic assembly ----->


def _notes():
    return [
        ResearchNote(
            id=uuid.uuid4(), user_id=USER_ID, sequence=1, task="ownership",
            content="Ownership content.", sources_used=["u1", "u2"],
        ),
        ResearchNote(
            id=uuid.uuid4(), user_id=USER_ID, sequence=2, task="borrows",
            content="Borrow content.", sources_used=["u3"],
        ),
    ]


async def test_editorial_plan_uses_the_submit_report_tool():
    model = ScriptedToolModel(
        responses=[
            tool_call(
                "SubmitReport",
                {
                    "sections": [
                        {"idx": 2, "include": True, "order": 1,
                         "lead_in": "Start with borrows:"},
                        {"idx": 1, "include": False, "order": 2},
                    ],
                    "intro": "Intro text.",
                },
            )
        ]
    )
    plan = await plan_note_report(model, "rust", _notes())
    assert plan.intro == "Intro text."
    assert plan.sections[0].idx == 2 and plan.sections[0].include


async def test_editorial_contract_violation_raises():
    model = ScriptedToolModel(responses=[AIMessage(content="prose")])
    with pytest.raises(Exception, match="SubmitReport"):
        await plan_note_report(model, "rust", _notes())


def test_assembly_is_deterministic_and_ordered():
    plan = SubmitReport.model_validate(
        {
            "sections": [
                {"idx": 2, "include": True, "order": 1,
                 "lead_in": "Start here:"},
                {"idx": 1, "include": True, "order": 2},
                {"idx": 3, "include": True, "order": 3},
            ],
            "intro": "Intro.",
        }
    )
    report = assemble_note_report(_notes(), plan)

    assert report.startswith("Intro.")
    assert report.index("Borrow content.") < report.index("Ownership content.")
    assert "Start here:" in report
    assert "## Sources" in report
    # borrows (idx 2) appears first, so u3 numbers ahead of u1/u2.
    assert "- [1] u3" in report and "- [2] u1" in report and "- [3] u2" in report
    # Unknown idx (3) is skipped, never invented.
    assert "None" not in report


# <----- research pipeline: notes + events ----->


async def test_research_pipeline_writes_notes_and_emits_events():
    repo = FakeNoteRepo()
    store = NoteStore(repo, USER_ID)
    model = ScriptedToolModel(
        responses=[tool_call("ResearchPlan", {"subquestions": ["rust"]})]
    )
    events = []
    async for event, data in research_logic(
        "rust",
        FakeSearchProvider(),
        model,
        FakeScraper(
            pages={"http://example.com/a": "Rust ownership is safe."}
        ),
        FakeRanker(),
        note_store=store,
    ):
        events.append((event, data))

    notes = [d for e, d in events if e == "note"]
    assert notes and notes[0]["idx"] == 1
    assert len(repo.rows) == 1  # the DB is the source of truth


async def test_research_pipeline_without_store_never_writes():
    model = ScriptedToolModel(
        responses=[tool_call("ResearchPlan", {"subquestions": ["rust"]})]
    )
    events = []
    async for event, data in research_logic(
        "rust",
        FakeSearchProvider(),
        model,
        FakeScraper(pages={"http://example.com/a": "Rust ownership."}),
        FakeRanker(),
    ):
        events.append((event, data))
    assert not [d for e, d in events if e == "note"]
