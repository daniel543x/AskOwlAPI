"""Offline tests for the research planner (no network, no DB, no models)."""

import pytest

from src.core.research.services.contracts import ResearchPlan
from src.core.research.services.planner import plan_subquestions
from tests.core_ask.fakes import ScriptedToolModel, tool_call


def planner_model(subquestions=None, responses=None):
    if responses is None:
        responses = [
            tool_call("ResearchPlan", {"subquestions": subquestions or []})
        ]
    return ScriptedToolModel(responses=responses)


@pytest.mark.asyncio
async def test_plan_happy_path_returns_subquestions():
    model = planner_model(["rust borrow checker", "rust memory safety"])
    subquestions, used_tool_call = await plan_subquestions(model, "rust safety", 5)

    assert subquestions == ["rust borrow checker", "rust memory safety"]
    assert used_tool_call is True


@pytest.mark.asyncio
async def test_plan_is_capped_at_max_subquestions():
    model = planner_model(["one", "two", "three", "four"])
    subquestions, _ = await plan_subquestions(model, "q", 2)

    assert subquestions == ["one", "two"]


@pytest.mark.asyncio
async def test_plan_strips_and_drops_empty_entries():
    model = planner_model(["  one  ", "", "   ", "two"])
    subquestions, _ = await plan_subquestions(model, "q", 5)

    assert subquestions == ["one", "two"]


@pytest.mark.asyncio
async def test_plan_tool_call_with_empty_list_falls_back_to_raw_query():
    model = planner_model([])
    subquestions, used_tool_call = await plan_subquestions(model, "raw query", 5)

    assert subquestions == ["raw query"]
    assert used_tool_call is False


@pytest.mark.asyncio
async def test_plan_without_tool_call_falls_back_to_raw_query():
    model = ScriptedToolModel(responses=[])
    from langchain_core.messages import AIMessage

    model.responses = [AIMessage(content="I would search for rust safety")]

    subquestions, used_tool_call = await plan_subquestions(model, "raw query", 5)

    assert subquestions == ["raw query"]
    assert used_tool_call is False


@pytest.mark.asyncio
async def test_plan_binds_the_contract_to_every_call():
    model = planner_model(["one"])
    await plan_subquestions(model, "q", 5)

    assert model.bindings
    for tools in model.bindings:
        names = [getattr(t, "__name__", None) for t in tools]
        assert ResearchPlan.__name__ in names
