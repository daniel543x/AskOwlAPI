"""Offline tests for the research pipeline adapters (no network, no DB)."""

import json
import re

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage

from src.core.research.services.pipeline import (
    research_json_adapter,
    research_sse_adapter,
)
from src.tools.settings import settings
from tests.core_ask.fakes import (
    FakeRanker,
    FakeScraper,
    FakeSearchProvider,
    ScriptedToolModel,
    make_scraper,
    tool_call,
)


@pytest.fixture(autouse=True)
def _sanity_filter_off(monkeypatch):
    """Synthetic sub-questions ("sq one") share no tokens with the fixture
    snippets, so the production sanity filter would drop every result and
    these tests would exercise the no-context path instead of the pipeline
    mechanics. Result hygiene has dedicated tests in
    tests/modules/test_sanity_filter.py and test_toolbox.py.
    """
    monkeypatch.setattr(settings, "RESULT_SANITY_FILTER", False)


def research_model(
    subquestions=None,
    report="## Section\nRust is memory safe. [Source](http://example.com/a)",
):
    return ScriptedToolModel(
        responses=[
            tool_call("ResearchPlan", {"subquestions": subquestions or ["sq one"]}),
            tool_call("ResearchReport", {"report": report}),
        ]
    )


def make_deps():
    return {
        "searching": FakeSearchProvider(),
        "scraper": make_scraper(),
        "ranker": FakeRanker(),
    }


def parse_frames(stream: list[str]) -> list[tuple[str, dict]]:
    parsed = []
    for frame in stream:
        match = re.match(r"event: (\w+)\nid: \d+\ndata: (.*)\n\n", frame, re.S)
        assert match, f"malformed SSE frame: {frame!r}"
        parsed.append((match.group(1), json.loads(match.group(2))))
    return parsed


@pytest.mark.asyncio
async def test_sse_happy_path_event_sequence():
    model = research_model(subquestions=["sq one", "sq two"])
    stream = [
        frame
        async for frame in research_sse_adapter("query", model=model, **make_deps())
    ]
    events = [name for name, _ in parse_frames(stream)]

    assert events[0] == "status"
    assert "plan" in events
    assert events.count("sources") == 2
    assert events.count("context") == 2
    assert "message" in events
    assert events[-1] == "done"
    assert "error" not in events


@pytest.mark.asyncio
async def test_sse_plan_event_carries_subquestions():
    model = research_model(subquestions=["alpha", "beta"])
    stream = [
        frame
        async for frame in research_sse_adapter("query", model=model, **make_deps())
    ]
    plan = next(data for name, data in parse_frames(stream) if name == "plan")

    assert plan["subquestions"] == ["alpha", "beta"]
    assert plan["tool_call"] is True


@pytest.mark.asyncio
async def test_sse_sources_and_context_carry_the_subquery():
    model = research_model(subquestions=["sq one"])
    stream = [
        frame
        async for frame in research_sse_adapter("query", model=model, **make_deps())
    ]
    parsed = parse_frames(stream)

    sources = next(d for n, d in parsed if n == "sources")
    context = next(d for n, d in parsed if n == "context")
    assert sources["subquery"] == "sq one"
    assert context["subquery"] == "sq one"
    assert context["chunks"] >= 1


@pytest.mark.asyncio
async def test_sse_frames_have_ids_framing_and_terminator():
    model = research_model()
    stream = [
        frame
        async for frame in research_sse_adapter("query", model=model, **make_deps())
    ]

    ids = []
    for frame in stream:
        match = re.search(r"id: (\d+)", frame)
        assert match is not None
        ids.append(int(match.group(1)))
    assert ids == sorted(ids) and ids[0] == 1
    assert all(frame.endswith("\n\n") for frame in stream)


@pytest.mark.asyncio
async def test_planner_fallback_runs_the_raw_query_as_single_subquestion():
    model = ScriptedToolModel(
        responses=[
            AIMessage(content="no tool call here"),
            tool_call("ResearchReport", {"report": "Report."}),
        ]
    )
    stream = [
        frame
        async for frame in research_sse_adapter("raw query", model=model, **make_deps())
    ]
    parsed = parse_frames(stream)

    plan = next(d for n, d in parsed if n == "plan")
    assert plan == {"subquestions": ["raw query"], "tool_call": False}
    # The gathering still ran for that single sub-question.
    assert any(n == "sources" for n, _ in parsed)


@pytest.mark.asyncio
async def test_no_context_yields_deterministic_report_without_extra_llm_call():
    deps = make_deps()
    deps["scraper"] = FakeScraper(return_empty=True)
    model = ScriptedToolModel(
        responses=[tool_call("ResearchPlan", {"subquestions": ["sq one"]})]
    )
    stream = [
        frame async for frame in research_sse_adapter("query", model=model, **deps)
    ]
    parsed = parse_frames(stream)

    assert len(model.calls) == 1  # planner only; no synthesis call
    done = next(d for n, d in parsed if n == "done")
    assert done == {"status": "no_context"}
    message = "".join(d["text"] for n, d in parsed if n == "message")
    assert "No relevant content" in message


@pytest.mark.asyncio
async def test_report_without_tool_call_is_error_event():
    model = ScriptedToolModel(
        responses=[
            tool_call("ResearchPlan", {"subquestions": ["sq one"]}),
            AIMessage(content="here is a report without a tool call"),
        ]
    )
    stream = [
        frame async for frame in research_sse_adapter(
            "query", model=model, **make_deps()
        )
    ]
    parsed = parse_frames(stream)

    errors = [d for n, d in parsed if n == "error"]
    assert errors and "ResearchReport" in errors[0]["error"]
    assert parsed[-1][0] == "error"


@pytest.mark.asyncio
async def test_pipeline_exception_becomes_error_event():
    class ExplodingSearch(FakeSearchProvider):
        async def search(self, query: str, engines: list[str] | None = None):
            raise RuntimeError("boom")

    deps = make_deps()
    deps["searching"] = ExplodingSearch()
    model = research_model()
    stream = [
        frame async for frame in research_sse_adapter("query", model=model, **deps)
    ]
    parsed = parse_frames(stream)

    errors = [d for n, d in parsed if n == "error"]
    assert errors and "boom" in errors[0]["error"]


@pytest.mark.asyncio
async def test_json_happy_path_returns_full_document():
    model = research_model(subquestions=["sq one"])
    result = await research_json_adapter("query", model=model, **make_deps())

    assert result["query"] == "query"
    assert result["subquestions"] == ["sq one"]
    assert len(result["gathered"]) == 1
    assert result["gathered"][0]["subquestion"] == "sq one"
    assert "Rust is memory safe" in result["report"]


@pytest.mark.asyncio
async def test_json_no_context_returns_deterministic_report():
    deps = make_deps()
    deps["scraper"] = FakeScraper(return_empty=True)
    model = ScriptedToolModel(
        responses=[tool_call("ResearchPlan", {"subquestions": ["sq one"]})]
    )
    result = await research_json_adapter("query", model=model, **deps)

    assert "No relevant content" in result["report"]


@pytest.mark.asyncio
async def test_json_contract_violation_is_502():
    model = ScriptedToolModel(
        responses=[
            tool_call("ResearchPlan", {"subquestions": ["sq one"]}),
            AIMessage(content="no tool call"),
        ]
    )
    with pytest.raises(HTTPException) as exc_info:
        await research_json_adapter("query", model=model, **make_deps())

    assert exc_info.value.status_code == 502


@pytest.mark.asyncio
async def test_every_llm_call_has_tools_bound():
    model = research_model(subquestions=["sq one", "sq two"])
    stream = [
        frame
        async for frame in research_sse_adapter("query", model=model, **make_deps())
    ]
    list(parse_frames(stream))

    assert model.bindings
    for tools in model.bindings:
        names = [getattr(t, "__name__", "") for t in tools]
        assert names  # never a bound-less LLM call


# <----- Reflect -> follow-up loop (WU-7 / WU-8) ----->


async def collect_research(model, **kwargs):
    deps = make_deps()
    deps.update(kwargs)
    stream = [
        frame
        async for frame in research_sse_adapter("query", model=model, **deps)
    ]
    return parse_frames(stream)


@pytest.mark.asyncio
async def test_loop_stays_single_pass_by_default():
    model = research_model(subquestions=["sq one", "sq two"])
    parsed = await collect_research(model)

    assert "reflect" not in [n for n, _ in parsed]
    assert parsed[-1] == ("done", {"status": "completed"})


@pytest.mark.asyncio
async def test_reflect_event_and_follow_ups_run_once():
    # Round 1 reflects, finds two missing angles, both are gathered, and the
    # round budget (max_iterations=2 -> one reflection cycle) stops the loop.
    model = ScriptedToolModel(
        responses=[
            tool_call("ResearchPlan", {"subquestions": ["sq one"]}),
            tool_call(
                "MissingInfo",
                {
                    "missing": ["benchmarks", "alternatives"],
                    "follow_up_queries": ["sq two", "sq three"],
                },
            ),
            tool_call("ResearchReport", {"report": "## Findings\nDone."}),
        ]
    )
    parsed = await collect_research(model, max_iterations=2)

    names = [n for n, _ in parsed]
    assert names.count("reflect") == 1
    reflect_payload = next(d for n, d in parsed if n == "reflect")
    assert reflect_payload["missing"] == ["benchmarks", "alternatives"]
    assert reflect_payload["rounds_used"] == 1
    assert names.count("sources") == 3  # sq one + two follow-ups
    assert [d["subquery"] for n, d in parsed if n == "context"] == [
        "sq one", "sq two", "sq three",
    ]
    assert parsed[-1] == ("done", {"status": "completed"})


@pytest.mark.asyncio
async def test_sufficient_context_ends_the_loop_after_one_reflection():
    model = ScriptedToolModel(
        responses=[
            tool_call("ResearchPlan", {"subquestions": ["sq one"]}),
            tool_call("MissingInfo", {"missing": [], "follow_up_queries": []}),
            tool_call("ResearchReport", {"report": "## Findings\nDone."}),
        ]
    )
    parsed = await collect_research(model, max_iterations=2)

    assert [n for n, _ in parsed].count("reflect") == 1
    assert [n for n, _ in parsed].count("sources") == 1  # no follow-up gathers


@pytest.mark.asyncio
async def test_max_iterations_clamped_to_settings_ceiling(monkeypatch):
    from src.tools.settings import settings

    monkeypatch.setattr(settings, "RESEARCH_MAX_ROUNDS", 1)
    model = research_model(subquestions=["sq one"])
    parsed = await collect_research(
        model,
        max_iterations=5,  # would request 4 reflection cycles
    )

    assert "reflect" not in [n for n, _ in parsed]


@pytest.mark.asyncio
async def test_budget_exhaustion_stops_loop_and_reports_reason(monkeypatch):
    from src.tools.settings import settings

    # Shrunken token budget: exhausted during/after the first gather, so
    # sub-question two and the reflection never run and the report degrades
    # to briefs-so-far with the budget done-status.
    monkeypatch.setattr(settings, "AGENT_TOKEN_BUDGET", 60)
    model = research_model(subquestions=["sq one", "sq two"])
    parsed = await collect_research(model, max_iterations=2)

    done = parsed[-1]
    assert done[0] == "done"
    assert done[1]["status"] == "budget_exhausted"
    assert "token" in done[1]["reason"]
    # Only the first sub-question was gathered before exhaustion.
    assert [d["subquery"] for n, d in parsed if n == "context"] == ["sq one"]


@pytest.mark.asyncio
async def test_compression_on_replaces_contexts_with_briefs(monkeypatch):
    from src.tools.settings import settings

    monkeypatch.setattr(settings, "RESEARCH_COMPRESS_CONTEXT", True)
    model = ScriptedToolModel(
        responses=[
            tool_call("ResearchPlan", {"subquestions": ["sq one"]}),
            tool_call("ContextBrief", {"brief": "Brief: 80.0% adoption."}),
            tool_call("ResearchReport", {"report": "## Section\nFindings."}),
        ]
    )
    result = await research_json_adapter(
        "query", model=model, **make_deps(), max_iterations=1
    )

    assert result["gathered"][0]["context"] == "Brief: 80.0% adoption."
    assert "first paragraph" not in result["gathered"][0]["context"]
    assert result["rounds_used"] == 0


# <----- Current date in research prompts (AR-7 task 1) ----->


@pytest.mark.asyncio
async def test_research_prompts_carry_the_current_date():
    model = research_model(subquestions=["sq one"])
    await collect_research(model, today="2020-01-01")

    # Planner (call 0) and synthesis (last call) both carry the date line.
    assert model.calls[0][0].startswith("Current date: 2020-01-01.")
    assert model.calls[-1][0].startswith("Current date: 2020-01-01.")
