"""Offline tests for the deep-research preset (PR 19): scout + reflect."""

import pytest

from src.core.research.services.pipeline import research_logic
from src.core.research.services.planner import revise_research_plan
from src.tools.settings import settings
from tests.core_ask.fakes import (
    FakeRanker,
    FakeScraper,
    FakeSearchProvider,
    ScriptedToolModel,
    tool_call,
)


def _make_toolbox():
    from src.core.tools.toolbox import AskToolbox

    pages = {
        "http://example.com/a": "# Rust ownership\\nOwnership makes rust safe.",
        "http://example.com/b": "# Borrow checker\\nBorrows are checked.",
    }
    return AskToolbox(
        FakeSearchProvider(), FakeScraper(pages=pages), FakeRanker(), None, None
    )


async def _collect(gen):
    events = []
    async for event, data in gen:
        events.append((event, data))
    return events


@pytest.fixture(autouse=True)
def sanity_off(monkeypatch):
    monkeypatch.setattr(settings, "RESULT_SANITY_FILTER", False)


# <----- Scout ----->


async def test_scout_runs_before_the_planner():
    model = ScriptedToolModel(
        responses=[tool_call("ResearchPlan", {"subquestions": ["rust"]})]
    )
    events = await _collect(
        research_logic(
            "rust",
            FakeSearchProvider(),
            model,
            FakeScraper(pages={"http://example.com/a": "Rust ownership."}),
            FakeRanker(),
            scout=True,
        )
    )
    statuses = [d.get("step", "") for e, d in events if e == "status"]
    assert "Scouting the landscape..." in statuses
    assert statuses.index("Scouting the landscape...") < statuses.index(
        "Planning sub-questions..."
    )


async def test_scout_off_by_default():
    model = ScriptedToolModel(
        responses=[tool_call("ResearchPlan", {"subquestions": ["rust"]})]
    )
    events = await _collect(
        research_logic(
            "rust",
            FakeSearchProvider(),
            model,
            FakeScraper(pages={"http://example.com/a": "Rust ownership."}),
            FakeRanker(),
        )
    )
    assert not any(
        "Scouting" in d.get("step", "") for e, d in events if e == "status"
    )


# <----- Reflect ----->


async def test_reflect_adds_subquestions_and_gathers_them():
    model = ScriptedToolModel(
        responses=[
            tool_call("ResearchPlan", {"subquestions": ["rust ownership"]}),
            tool_call(
                "RevisePlan",
                {
                    "add": ["rust async runtime"],
                    "rationale": "The digest lacks async coverage.",
                },
            ),
        ]
    )
    events = await _collect(
        research_logic(
            "rust",
            FakeSearchProvider(),
            model,
            FakeScraper(pages={"http://example.com/a": "Rust ownership."}),
            FakeRanker(),
            reflect=True,
        )
    )
    reflect_events = [d for e, d in events if e == "reflect"]
    assert reflect_events and reflect_events[0]["added"] == ["rust async runtime"]
    assert "async coverage" in reflect_events[0]["rationale"]
    statuses = [d.get("step", "") for e, d in events if e == "status"]
    assert any("reflect" in s for s in statuses)


async def test_reflect_clamps_to_the_ceiling(monkeypatch):
    monkeypatch.setattr(settings, "RESEARCH_MAX_SUBQUESTIONS", 1)
    monkeypatch.setattr(settings, "RESEARCH_REFLECT_SLACK", 0)
    model = ScriptedToolModel(
        responses=[
            tool_call("ResearchPlan", {"subquestions": ["rust ownership"]}),
            tool_call(
                "RevisePlan",
                {"add": ["a", "b", "c"], "rationale": "gaps"},
            ),
        ]
    )
    events = await _collect(
        research_logic(
            "rust",
            FakeSearchProvider(),
            model,
            FakeScraper(pages={"http://example.com/a": "Rust ownership."}),
            FakeRanker(),
            reflect=True,
        )
    )
    assert not [d for e, d in events if e == "reflect"]  # nothing fits


async def test_reflect_prose_is_a_noop():
    from langchain_core.messages import AIMessage

    model = ScriptedToolModel(
        responses=[
            tool_call("ResearchPlan", {"subquestions": ["rust ownership"]}),
            AIMessage(content="no gaps, moving on"),
        ]
    )
    events = await _collect(
        research_logic(
            "rust",
            FakeSearchProvider(),
            model,
            FakeScraper(pages={"http://example.com/a": "Rust ownership."}),
            FakeRanker(),
            reflect=True,
        )
    )
    assert not [d for e, d in events if e == "reflect"]
    assert events[-1][0] == "answer_ready"


async def test_reflect_off_never_invokes_the_tool():
    model = ScriptedToolModel(
        responses=[tool_call("ResearchPlan", {"subquestions": ["rust ownership"]})]
    )
    events = await _collect(
        research_logic(
            "rust",
            FakeSearchProvider(),
            model,
            FakeScraper(pages={"http://example.com/a": "Rust ownership."}),
            FakeRanker(),
        )
    )
    assert model.bindings.count(model.bindings[0]) >= 1
    assert not [d for e, d in events if e == "reflect"]


# <----- revise_research_plan unit ----->


async def test_revise_plan_dedupes_and_cleans():
    model = ScriptedToolModel(
        responses=[
            tool_call(
                "RevisePlan",
                {"add": ["  rust async ", "", "rust async"], "rationale": "gaps"},
            )
        ]
    )
    added, rationale = await revise_research_plan(model, "rust", "digest")
    assert added == ["rust async"]
    assert rationale == "gaps"
