"""Pipeline-level tests for pre-score surfacing (offline)."""

import json
import re

import pytest

from src.core.ask.services.search import search_json_adapter, search_sse_adapter
from src.tools.settings import settings
from tests.core_ask.fakes import (
    FakeRanker,
    FakeSearchProvider,
    ScriptedToolModel,
    make_scraper,
    tool_call,
)
from tests.core_tools.test_pre_scores import FakeDecisionMaker


def make_deps():
    return {
        "searching": FakeSearchProvider(),
        "scraper": make_scraper(),
        "ranker": FakeRanker(),
    }


def parse_frames(stream):
    parsed = []
    for frame in stream:
        match = re.match(r"event: (\w+)\nid: \d+\ndata: (.*)\n\n", frame, re.S)
        assert match, f"malformed frame: {frame!r}"
        parsed.append((match.group(1), json.loads(match.group(2))))
    return parsed


@pytest.mark.asyncio
async def test_grounding_event_in_sse_stream(monkeypatch):
    monkeypatch.setattr(settings, "PRE_SCORE_GROUNDING_CHECK", True)
    model = ScriptedToolModel(
        responses=[
            tool_call("SubmitSearchQuery", {"query": "rust"}),
            tool_call("CitedAnswer", {"answer": "Answer. [Source](u)"}),
        ]
    )
    maker = FakeDecisionMaker(noul_yes=0.87)

    stream = [
        frame
        async for frame in search_sse_adapter(
            "rust", model=model, decision_maker=maker, **make_deps()
        )
    ]
    parsed = parse_frames(stream)

    grounding = [d for name, d in parsed if name == "grounding"]
    assert grounding == [{"supported": pytest.approx(0.87)}]
    # Surfaced, never blocking: done still completes the stream.
    assert parsed[-1][0] == "done"
    assert parsed[-1][1] == {"status": "completed"}


@pytest.mark.asyncio
async def test_grounding_absent_when_toggle_off():
    model = ScriptedToolModel(
        responses=[
            tool_call("SubmitSearchQuery", {"query": "rust"}),
            tool_call("CitedAnswer", {"answer": "A"}),
        ]
    )
    maker = FakeDecisionMaker(noul_yes=0.87)

    stream = [
        frame
        async for frame in search_sse_adapter(
            "rust", model=model, decision_maker=maker, **make_deps()
        )
    ]
    assert not [f for f in stream if "event: grounding" in f]


@pytest.mark.asyncio
async def test_grounding_field_in_json_response(monkeypatch):
    monkeypatch.setattr(settings, "PRE_SCORE_GROUNDING_CHECK", True)
    model = ScriptedToolModel(
        responses=[
            tool_call("SubmitSearchQuery", {"query": "rust"}),
            tool_call("CitedAnswer", {"answer": "A. [Source](u)"}),
        ]
    )

    result = await search_json_adapter(
        "rust", model=model, decision_maker=FakeDecisionMaker(noul_yes=0.64),
        **make_deps(),
    )

    assert result["grounding_supported"] == pytest.approx(0.64)


@pytest.mark.asyncio
async def test_query_choice_pre_score_picks_raw(monkeypatch):
    monkeypatch.setattr(settings, "PRE_SCORE_QUERY_CHOICE", True)
    model = ScriptedToolModel(
        responses=[
            tool_call("SubmitSearchQuery", {"query": "optimized query"}),
            tool_call("CitedAnswer", {"answer": "A"}),
        ]
    )
    maker = FakeDecisionMaker(choice="raw")

    stream = [
        frame
        async for frame in search_sse_adapter(
            "raw query", model=model, decision_maker=maker, **make_deps()
        )
    ]
    query_events = [d for name, d in parse_frames(stream) if name == "query"]

    assert query_events[0]["query"] == "raw query"
