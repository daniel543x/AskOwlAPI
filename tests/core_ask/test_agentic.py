"""Offline tests for the agentic search core (PR 17 / decision 8)."""

import pytest

from src.core.ask.agentic import (
    AgentDegraded,
    agentic_search_logic,
    search_budget,
)
from src.core.ask.services.search import search_logic
from src.tools.settings import settings
from tests.core_ask.fakes import (
    FakeRanker,
    FakeScraper,
    FakeSearchProvider,
    ScriptedToolModel,
    tool_call,
)


def _make_toolbox(**overrides):
    from src.core.tools.toolbox import AskToolbox

    pages = {
        "http://example.com/a": "# Rust ownership\nOwnership makes rust safe.",
        "http://example.com/b": "# Rust borrow checker\nBorrows are checked.",
    }
    return AskToolbox(
        FakeSearchProvider(),
        FakeScraper(pages=pages),
        FakeRanker(),
        None,
        None,
        **overrides,
    )


async def _collect(gen):
    events = []
    async for event, data in gen:
        events.append((event, data))
    return events


@pytest.fixture(autouse=True)
def agentic_flag(monkeypatch):
    monkeypatch.setattr(settings, "AGENTIC_SEARCH", False)
    monkeypatch.setattr(settings, "RESULT_SANITY_FILTER", False)


def budget(iterations: int = 3):
    from src.core.research.services.budgets import Budget

    return Budget(
        max_iterations=iterations, time_budget_seconds=30.0, token_budget=10_000
    )


# <----- The core loop ----->


async def test_loop_gathers_then_streams_the_cited_answer():
    model = ScriptedToolModel(
        responses=[
            tool_call("web_search", {"query": "rust ownership"}),
            tool_call("CitedAnswer", {"answer": "Answer. [Source](u)"}),
        ]
    )
    events = await _collect(
        agentic_search_logic("rust", model, _make_toolbox(), budget())
    )

    kinds = [event for event, _ in events]
    assert "status" in kinds and "sources" in kinds
    assert "message" in kinds  # streamed answer deltas
    final = events[-1][1]
    assert events[-1][0] == "agentic_answer"
    assert final["answer"] == "Answer. [Source](u)"
    assert final["sources"], "gathered sources must surface"


async def test_contract_violation_repairs_once_then_answers():
    model = ScriptedToolModel(
        responses=[
            __import__("langchain_core.messages", fromlist=["AIMessage"]).AIMessage(
                content="prose without a tool call"
            ),
            tool_call("CitedAnswer", {"answer": "Recovered. [Source](u)"}),
        ]
    )
    events = await _collect(
        agentic_search_logic("rust", model, _make_toolbox(), budget())
    )
    assert events[-1][0] == "agentic_answer"


async def test_contract_violation_after_repair_degrades():
    from langchain_core.messages import AIMessage

    model = ScriptedToolModel(
        responses=[AIMessage(content="prose one"), AIMessage(content="prose two")]
    )
    with pytest.raises(AgentDegraded, match="tool-contract violation"):
        await _collect(agentic_search_logic("rust", model, _make_toolbox(), budget()))


async def test_budget_exhaustion_without_material_degrades():
    model = ScriptedToolModel(
        responses=[tool_call("web_search", {"query": "rust"})],
    )
    with pytest.raises(AgentDegraded, match="without material"):
        await _collect(
            agentic_search_logic("rust", model, _make_toolbox(), budget(1))
        )


async def test_budget_exhaustion_with_material_forces_synthesis():
    model = ScriptedToolModel(
        responses=[
            tool_call("web_search", {"query": "rust"}),
            tool_call("fetch_pages", {}),
            tool_call("CitedAnswer", {"answer": "Forced. [Source](u)"}),
        ]
    )
    events = await _collect(
        agentic_search_logic("rust", model, _make_toolbox(), budget(2))
    )
    assert events[-1][0] == "agentic_answer"
    assert events[-1][1]["answer"] == "Forced. [Source](u)"


def test_search_budget_preset_is_low():
    b = search_budget()
    assert 1 <= b.max_iterations <= 3


# <----- Pipeline integration: flag + degrade ----->


async def test_flag_off_runs_the_deterministic_pipeline(monkeypatch):
    monkeypatch.setattr(settings, "AGENTIC_SEARCH", False)
    model = ScriptedToolModel(
        responses=[
            tool_call("SubmitSearchQuery", {"query": "rust"}),
            tool_call("CitedAnswer", {"answer": "Det. [Source](u)"}),
        ]
    )
    events = await _collect(
        search_logic(
            "rust",
            FakeSearchProvider(),
            model,
            FakeScraper(),
            FakeRanker(),
        )
    )
    kinds = [event for event, _ in events]
    assert "degraded" not in kinds
    assert "agentic_answer" not in kinds


async def test_flag_on_runs_the_core_without_rewrite(monkeypatch):
    monkeypatch.setattr(settings, "AGENTIC_SEARCH", True)
    model = ScriptedToolModel(
        responses=[
            tool_call("web_search", {"query": "rust"}),
            tool_call("CitedAnswer", {"answer": "Agentic. [Source](u)"}),
        ]
    )
    events = await _collect(
        search_logic(
            "rust",
            FakeSearchProvider(),
            model,
            FakeScraper(),
            FakeRanker(),
        )
    )
    kinds = [event for event, _ in events]
    assert "agentic_answer" in kinds
    # The core replaces the rewrite step: no query_optimization stage.
    statuses = [d.get("step", "") for e, d in events if e == "status"]
    assert not any("Building query" in s for s in statuses)


async def test_flag_on_degrades_into_the_deterministic_pipeline(monkeypatch):
    from langchain_core.messages import AIMessage

    monkeypatch.setattr(settings, "AGENTIC_SEARCH", True)
    model = ScriptedToolModel(
        responses=[
            AIMessage(content="prose one"),
            AIMessage(content="prose two"),
            tool_call("SubmitSearchQuery", {"query": "rust"}),
            tool_call("CitedAnswer", {"answer": "Det. [Source](u)"}),
        ]
    )
    events = await _collect(
        search_logic(
            "rust",
            FakeSearchProvider(),
            model,
            FakeScraper(),
            FakeRanker(),
        )
    )
    kinds = [event for event, _ in events]
    assert "degraded" in kinds
    assert kinds[-1] == "answer_ready"  # deterministic fallback completed
    degraded = next(d for e, d in events if e == "degraded")
    assert "tool-contract" in degraded["reason"]
