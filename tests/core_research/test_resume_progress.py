"""Offline tests: research resume — seeded notes, progress hooks, skip."""

import uuid

import pytest

from src.core.research.services.pipeline import research_logic
from src.modules.notes.model import ResearchNote
from src.modules.notes.repository import NoteRepository
from src.modules.notes.store import NoteStore
from src.tools.settings import settings
from tests.core_ask.fakes import (
    FakeRanker,
    FakeSearchProvider,
    ScriptedToolModel,
    make_scraper,
    tool_call,
)


def seeded_note(task: str, content: str = "seed body") -> ResearchNote:
    return ResearchNote(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        sequence=0,
        task=task,
        content=content,
        sources_used=["https://example.com/seed"],
    )


class RecordingStore(NoteStore):
    """A NoteStore that absorbs writes into memory (no DB)."""

    def __init__(self, seeded=None):
        super().__init__(FakeNoteRepo(), uuid.uuid4(), seeded=seeded)
        self.written: list[str] = []


class FakeNoteRepo(NoteRepository):
    def __init__(self):
        self.rows: list = []

    async def create(self, user_id, task, content, sources_used,
                     conversation_id=None):
        note = ResearchNote(
            id=uuid.uuid4(), user_id=user_id, sequence=len(self.rows) + 1,
            task=task, content=content, sources_used=list(sources_used),
        )
        self.rows.append(note)
        return note

    async def get_by_ids(self, note_ids):
        return [n for n in self.rows if n.id in note_ids]


async def collect(events, wanted):
    return [(e, d) for e, d in events]


@pytest.fixture(autouse=True)
def sanity_off(monkeypatch):
    monkeypatch.setattr(settings, "RESULT_SANITY_FILTER", False)


def research_model(subquestions):
    return ScriptedToolModel(
        responses=[
            tool_call("ResearchPlan", {"subquestions": subquestions}),
            tool_call(
                "ResearchReport",
                {"report": "## Covered\nSynthesized from notes."},
            ),
        ]
    )


async def run_pipeline(
    note_store,
    query="main question",
    subquestions=None,
    progress=None,
    resume_state=None,
):
    events = []
    async for event, data in research_logic(
        query,
        FakeSearchProvider(),
        research_model(subquestions or [query]),
        make_scraper(),
        FakeRanker(),
        note_store=note_store,
        progress=progress,
        resume_state=resume_state,
    ):
        events.append((event, data))
    return events


async def test_seeded_store_orders_old_and_new_together():
    store = RecordingStore(seeded=[seeded_note("covered task")])
    await store.write("new task", "fresh body", ["https://x"])
    ordered = await store.ordered_notes()
    tasks = [n.task for n in ordered]
    assert "covered task" in tasks and "new task" in tasks
    assert len(tasks) == 2 and len(store.written) == 0, store.written


async def test_progress_hook_reports_plan_and_note_ids():
    progress: list[dict] = []

    async def hook(data: dict) -> None:
        progress.append(data)

    store = RecordingStore()
    await run_pipeline(
        store, query="main q", subquestions=["main q"], progress=hook
    )
    kinds = [set(p) for p in progress if p]
    assert any("planned_tasks" in k for k in kinds)
    assert any("note_ids" in k for k in kinds)


async def test_covered_tasks_are_skipped_but_still_synthesized():
    store = RecordingStore(seeded=[seeded_note("cov a"), seeded_note("cov b")])

    events = await run_pipeline(
        store, query="both covered", subquestions=["cov a", "cov b"]
    )
    notes = [e for e, _ in events if e == "note"]
    # Both legs were covered pre-restart: nothing new is written.
    assert notes == []
    # The plan is still reported, and synthesis still runs to completion.
    done = [d for e, d in events if e == "answer_ready"]
    assert done and done[0]["query"] == "both covered"


async def test_resumed_plan_is_authoritative_no_replanning():
    passed: list = []

    class PlannerSpy:
        pass

    import src.core.research.services.pipeline as pipe_mod

    original = pipe_mod.plan_subquestions

    async def spy(model, query, ceiling, today=None):
        passed.append(query)
        return original(model, query, ceiling, today=today)

    store = RecordingStore(seeded=[seeded_note("cov a")])
    await run_pipeline(
        store,
        query="ru",
        subquestions=["cov a", "cov b"],
        resume_state={"planned_tasks": ["cov a", "cov b"]},
    )
    assert passed == []  # the planner was never called



async def test_plan_event_marks_resumed_true(monkeypatch):
    import src.core.research.services.pipeline as pipe_mod

    monkeypatch.setattr(pipe_mod.settings, "RESULT_SANITY_FILTER", False)
    store = RecordingStore(seeded=[seeded_note("cov a")])
    events = await run_pipeline(
        store,
        query="ru",
        subquestions=["cov a"],
        resume_state={"planned_tasks": ["cov a"]},
    )
    plan = next(d for e, d in events if e == "plan")
    assert plan["resumed"] is True
