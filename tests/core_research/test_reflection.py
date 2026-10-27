"""Offline tests for the research reflection step (WU-7 / AR-3)."""

from langchain_core.messages import AIMessage

from src.core.research.services.reflection import reflect
from tests.core_ask.fakes import ScriptedToolModel, tool_call

GATHERED = [
    {"subquestion": "rust ownership", "context": "Ownership moves values."},
    {"subquestion": "rust lifetimes", "context": "Lifetimes annotate references."},
]


async def test_reflection_ends_when_sufficient():
    model = ScriptedToolModel(
        responses=[
            tool_call("MissingInfo", {"missing": [], "follow_up_queries": []})
        ]
    )

    result = await reflect(model, "explain rust", GATHERED)

    assert result.missing == []
    assert result.follow_up_queries == []
    # The report (question + gathered sections) reached the model.
    prompt = model.calls[0][1]
    assert "explain rust" in prompt
    assert "rust ownership" in prompt and "Ownership moves values." in prompt


async def test_reflection_parses_missing_and_follow_ups():
    model = ScriptedToolModel(
        responses=[
            tool_call(
                "MissingInfo",
                {
                    "missing": ["concrete benchmarks"],
                    "follow_up_queries": ["rust ownership benchmarks"],
                },
            )
        ]
    )

    result = await reflect(model, "explain rust", GATHERED)

    assert result.missing == ["concrete benchmarks"]
    assert result.follow_up_queries == ["rust ownership benchmarks"]


async def test_reflection_blank_entries_are_dropped():
    model = ScriptedToolModel(
        responses=[
            tool_call(
                "MissingInfo",
                {"missing": ["  "], "follow_up_queries": ["", "real query"]},
            )
        ]
    )

    result = await reflect(model, "explain rust", GATHERED)

    assert result.missing == []
    assert result.follow_up_queries == ["real query"]


async def test_reflection_tool_error_stops_deterministically():
    model = ScriptedToolModel(responses=[AIMessage(content="prose, no tool call")])

    result = await reflect(model, "explain rust", GATHERED)

    assert result.missing == []
    assert result.follow_up_queries == []
