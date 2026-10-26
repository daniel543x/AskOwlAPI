"""Offline tests for research context compression (WU-8 / AR-3)."""

from langchain_core.messages import AIMessage

from src.core.research.services.compression import compress
from tests.core_ask.fakes import ScriptedToolModel, tool_call


async def test_compression_preserves_numbers():
    model = ScriptedToolModel(
        responses=[
            tool_call(
                "ContextBrief",
                {"brief": "Adoption reached 80.0% in 2026, up from 61.3%."},
            )
        ]
    )

    brief = await compress(
        model,
        "rust adoption",
        "",
        "Survey says adoption reached 80.0% in 2026, up from 61.3%.",
        2000,
    )

    assert "80.0%" in brief
    assert "61.3%" in brief


async def test_first_iteration_uses_create_prompt():
    model = ScriptedToolModel(
        responses=[tool_call("ContextBrief", {"brief": "b"})]
    )

    await compress(model, "rust adoption", "", "raw context", 2000)

    prompt = model.calls[0][1]
    assert "Create a summary" in prompt
    assert "rust adoption" in prompt
    assert "raw context" in prompt


async def test_update_prompt_merges_brief_and_new_context():
    model = ScriptedToolModel(
        responses=[tool_call("ContextBrief", {"brief": "updated"})]
    )

    brief = await compress(
        model, "rust adoption", "existing brief text", "new facts", 2000
    )

    assert brief == "updated"
    prompt = model.calls[0][1]
    assert "Update the existing running brief" in prompt
    assert "existing brief text" in prompt
    assert "new facts" in prompt


async def test_compression_caps_input_at_token_budget():
    model = ScriptedToolModel(
        responses=[tool_call("ContextBrief", {"brief": "b"})]
    )
    huge_context = "x" * 40_000  # ~10k estimated tokens

    await compress(model, "rust adoption", "", huge_context, 100)

    prompt = model.calls[0][1]
    # The context is capped near budget * 4 chars before the call.
    assert len(prompt) < 100 * 4 + 500
    assert huge_context not in prompt


async def test_compression_degrades_to_concatenation_on_tool_error():
    model = ScriptedToolModel(responses=[AIMessage(content="prose, no tool")])

    brief = await compress(model, "q", "old brief", "new context", 2000)

    assert "old brief" in brief and "new context" in brief
