"""Offline tests for the agent loop (no network, no DB, no models)."""


import pytest
from langchain_core.messages import AIMessage

from src.core.research.services.agent import agent_logic
from src.core.research.services.budgets import Budget
from src.core.tools.toolbox import AskToolbox
from tests.core_ask.fakes import (
    FakeRanker,
    FakeSearchProvider,
    ScriptedToolModel,
    make_scraper,
    tool_call,
)


def make_toolbox() -> AskToolbox:
    return AskToolbox(
        FakeSearchProvider(), make_scraper(), FakeRanker(), embedder=None
    )


def big_budget() -> Budget:
    return Budget(max_iterations=8, time_budget_seconds=60.0, token_budget=100_000)


async def collect(gen):
    return [event async for event in gen]


def frames_by_event(events, name):
    return [data for event, data in events if event == name]


@pytest.mark.asyncio
async def test_agent_happy_path_calls_tools_then_outcome():
    model = ScriptedToolModel(
        responses=[
            tool_call("web_search", {"query": "rust safety"}),
            tool_call("fetch_pages", {"urls": None}),
            tool_call("get_context", {"query": "rust safety"}),
            tool_call(
                "SubmitOutcome",
                {"answer": "Rust is safe. [Source](http://example.com/a)"},
            ),
        ]
    )
    events = await collect(agent_logic("goal", model, make_toolbox(), big_budget()))

    statuses = [d["step"] for d in frames_by_event(events, "status")]
    assert any("web_search" in s for s in statuses)
    assert any("fetch_pages" in s for s in statuses)
    assert any("get_context" in s for s in statuses)

    messages = [m["text"] for m in frames_by_event(events, "message")]
    assert messages == ["Rust is safe. [Source](http://example.com/a)"]

    done = frames_by_event(events, "done")
    assert done == [{"status": "completed"}]
    assert not frames_by_event(events, "error")


@pytest.mark.asyncio
async def test_agent_neever_accepts_free_form_responses():
    model = ScriptedToolModel(
        responses=[AIMessage(content="I think rust is safe.")]
    )
    events = await collect(agent_logic("goal", model, make_toolbox(), big_budget()))

    errors = frames_by_event(events, "error")
    assert errors and "free-form" in errors[0]["error"]
    assert not frames_by_event(events, "done")


@pytest.mark.asyncio
async def test_agent_empty_outcome_is_an_error():
    model = ScriptedToolModel(
        responses=[tool_call("SubmitOutcome", {"answer": "   "})]
    )
    events = await collect(agent_logic("goal", model, make_toolbox(), big_budget()))

    errors = frames_by_event(events, "error")
    assert errors and "empty answer" in errors[0]["error"]


@pytest.mark.asyncio
async def test_agent_budget_exhausted_with_context_forces_synthesis():
    # Iteration budget of 1: one round executing all three tools, then the
    # loop stops and synthesis is forced from the gathered context.
    budget = Budget(max_iterations=1, time_budget_seconds=60.0, token_budget=100_000)
    gather_all = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "web_search",
                "args": {"query": "rust"},
                "id": "c1",
                "type": "tool_call",
            },
            {
                "name": "fetch_pages",
                "args": {"urls": None},
                "id": "c2",
                "type": "tool_call",
            },
            {
                "name": "get_context",
                "args": {"query": "rust"},
                "id": "c3",
                "type": "tool_call",
            },
        ],
    )
    model = ScriptedToolModel(
        responses=[
            gather_all,
            tool_call("SubmitOutcome", {"answer": "Partial answer."}),
        ]
    )
    events = await collect(agent_logic("goal", model, make_toolbox(), budget))

    done = frames_by_event(events, "done")
    assert done[0]["status"] == "budget_exhausted"
    assert "iteration budget" in done[0]["reason"]

    messages = "".join(m["text"] for m in frames_by_event(events, "message"))
    assert "Partial answer." in messages
    assert not frames_by_event(events, "error")


@pytest.mark.asyncio
async def test_agent_budget_exhausted_without_context_is_deterministic():
    budget = Budget(max_iterations=0, time_budget_seconds=60.0, token_budget=100_000)
    model = ScriptedToolModel(responses=[])
    events = await collect(agent_logic("goal", model, make_toolbox(), budget))

    messages = "".join(m["text"] for m in frames_by_event(events, "message"))
    assert "stopped early" in messages
    done = frames_by_event(events, "done")
    assert done[0]["status"] == "budget_exhausted"


@pytest.mark.asyncio
async def test_agent_time_budget_stops_the_loop():
    budget = Budget(max_iterations=8, time_budget_seconds=0.0, token_budget=100_000)
    model = ScriptedToolModel(
        responses=[tool_call("web_search", {"query": "rust"})]
    )
    events = await collect(agent_logic("goal", model, make_toolbox(), budget))

    # Loop never ran; no context gathered; deterministic notice.
    assert budget.exhausted
    done = frames_by_event(events, "done")
    assert done[0]["status"] == "budget_exhausted"
    assert "time budget" in done[0]["reason"]


@pytest.mark.asyncio
async def test_agent_unknown_tool_returns_error_string_not_crash():
    model = ScriptedToolModel(
        responses=[
            tool_call("does_not_exist", {"x": 1}),
            tool_call("SubmitOutcome", {"answer": "Done anyway."}),
        ]
    )
    events = await collect(agent_logic("goal", model, make_toolbox(), big_budget()))

    assert frames_by_event(events, "done") == [{"status": "completed"}]


@pytest.mark.asyncio
async def test_agent_every_llm_call_has_tools_bound():
    model = ScriptedToolModel(
        responses=[
            tool_call("web_search", {"query": "rust"}),
            tool_call("SubmitOutcome", {"answer": "Answer."}),
        ]
    )
    await collect(agent_logic("goal", model, make_toolbox(), big_budget()))

    assert model.bindings
    for tools in model.bindings:
        assert len(tools) >= 1  # executable tools + SubmitOutcome


@pytest.mark.asyncio
async def test_agent_tool_exception_becomes_error_event():
    class ExplodingSearch(FakeSearchProvider):
        async def search(self, query: str):
            raise RuntimeError("search backend down")

    toolbox = AskToolbox(ExplodingSearch(), make_scraper(), FakeRanker())
    model = ScriptedToolModel(
        responses=[tool_call("web_search", {"query": "rust"})]
    )
    events = await collect(agent_logic("goal", model, toolbox, big_budget()))

    errors = frames_by_event(events, "error")
    assert errors and "search backend down" in errors[0]["error"]


def test_budget_token_estimation():
    budget = big_budget()
    budget.spend_tokens("x" * 400)  # ~100 tokens
    assert budget.tokens_spent == 100
    assert not budget.tokens_exhausted

    budget.spend_tokens("y" * 400_000)
    assert budget.tokens_exhausted
    assert budget.stop_reason() == "token budget exhausted"
