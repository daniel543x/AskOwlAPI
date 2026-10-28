import json

import pytest
from langchain_core.messages import AIMessage

from src.core.ask.services.contracts import CitedAnswer, SubmitSearchQuery
from src.core.ask.services.llm_steps import (
    ToolArgStream,
    ToolProtocolError,
    generate_cited_answer,
    rewrite_query,
    stream_cited_answer,
)
from tests.core_ask.fakes import ScriptedToolModel, tool_call

# <----- rewrite_query ----->


async def test_rewrite_query_uses_submit_search_query_tool_call():
    model = ScriptedToolModel(
        responses=[tool_call("SubmitSearchQuery", {"query": "rust lang"})]
    )

    optimized, used_tool_call = await rewrite_query(model, "what is rust?")

    assert optimized == "rust lang"
    assert used_tool_call is True
    assert model.bindings == [[SubmitSearchQuery]]
    assert len(model.calls) == 1


async def test_rewrite_query_falls_back_to_raw_query_without_tool_call():
    model = ScriptedToolModel(responses=[AIMessage(content="just prose, no tool")])

    optimized, used_tool_call = await rewrite_query(model, "original question")

    assert optimized == "original question"
    assert used_tool_call is False


async def test_rewrite_query_ignores_other_tool_calls():
    model = ScriptedToolModel(
        responses=[tool_call("SomeOtherTool", {"query": "nope"})]
    )

    optimized, used_tool_call = await rewrite_query(model, "raw q")

    assert optimized == "raw q"
    assert used_tool_call is False


# <----- generate_cited_answer ----->


async def test_generate_cited_answer_reads_tool_args():
    model = ScriptedToolModel(
        responses=[tool_call("CitedAnswer", {"answer": "The answer with [Source](u)."})]
    )

    answer = await generate_cited_answer(model, "q", "some context")

    assert answer == "The answer with [Source](u)."
    assert model.bindings == [[CitedAnswer]]
    assert "some context" in model.calls[0][0]


async def test_generate_cited_answer_rejects_free_form_output():
    model = ScriptedToolModel(responses=[AIMessage(content="free-form answer")])

    with pytest.raises(ToolProtocolError, match="CitedAnswer"):
        await generate_cited_answer(model, "q", "context")


# <----- stream_cited_answer ----->


async def test_stream_cited_answer_yields_answer_deltas():
    answer_text = "Rust is fast and safe. [Source](http://a)"
    model = ScriptedToolModel(
        responses=[tool_call("CitedAnswer", {"answer": answer_text})],
        stream_piece_size=7,
    )

    deltas = [delta async for _, delta in stream_cited_answer(model, "q", "ctx")]

    assert model.bindings == [[CitedAnswer]]
    assert "".join(deltas) == answer_text
    assert len(deltas) > 1  # actually streamed in pieces


async def test_stream_cited_answer_decodes_escapes():
    answer_text = 'line1\nline2 "quoted" back\\slash caf\u00e9'
    model = ScriptedToolModel(
        responses=[tool_call("CitedAnswer", {"answer": answer_text})],
        stream_piece_size=5,  # splits inside escape sequences
    )

    result = "".join(
        [delta async for _, delta in stream_cited_answer(model, "q", "ctx")]
    )

    assert result == answer_text


async def test_stream_cited_answer_rejects_content_without_tool_call():
    model = ScriptedToolModel(responses=[AIMessage(content="not a tool call")])

    with pytest.raises(ToolProtocolError, match="CitedAnswer"):
        async for _ in stream_cited_answer(model, "q", "ctx"):
            pass


async def test_stream_cited_answer_tags_reasoning_deltas():
    answer_text = "Final answer."
    model = ScriptedToolModel(
        responses=[tool_call("CitedAnswer", {"answer": answer_text})],
        stream_piece_size=4,
        reasoning_prefix="thinking hard",
    )

    tagged = [
        (kind, delta) async for kind, delta in stream_cited_answer(model, "q", "ctx")
    ]

    assert tagged[0] == ("thinking", "thinking hard")
    assert "".join(delta for kind, delta in tagged if kind == "answer") == answer_text
    assert all(kind == "answer" for kind, _ in tagged[1:])


# <----- ToolArgStream ----->


def feed_all(key: str, args_json: str, piece_size: int = 3) -> str:
    stream = ToolArgStream(key=key)
    out = ""
    for i in range(0, len(args_json), piece_size):
        out += stream.feed(args_json[i : i + piece_size])
    out += stream.finish()
    return out


def test_tool_arg_stream_simple_value():
    assert feed_all("answer", '{"answer": "hello world"}') == "hello world"


def test_tool_arg_stream_with_other_keys_before():
    args = json.dumps({"foo": "bar", "answer": "target"})
    assert feed_all("answer", args) == "target"


def test_tool_arg_stream_whitespace_between_tokens():
    assert feed_all("answer", '{ "answer" : "v" }') == "v"


def test_tool_arg_stream_escapes():
    value = feed_all("answer", '{"answer": "a\\nb\\tc\\"d\\\\e"}')
    assert value == 'a\nb\tc"d\\e'


def test_tool_arg_stream_unicode_escape():
    assert feed_all("answer", '{"answer": "caf\\u00e9"}') == "café"


def test_tool_arg_stream_unterminated_string_flushes():
    stream = ToolArgStream(key="answer")
    out = stream.feed('{"answer": "partial')
    out += stream.finish()
    assert out == "partial"
    assert stream.saw_value is True


def test_tool_arg_stream_missing_value():
    stream = ToolArgStream(key="answer")
    stream.feed('{"other": "x"}')
    stream.finish()
    assert stream.saw_value is False


def test_tool_arg_stream_empty_value():
    assert feed_all("answer", '{"answer": ""}') == ""


def test_tool_arg_stream_key_prefix_does_not_match():
    # "annotated" must not satisfy the "answer" key match.
    stream = ToolArgStream(key="answer")
    stream.feed('{"annotated": "nope", "answer": "yes"}')
    stream.finish()
    assert stream.saw_value is True


# <----- Current date in prompts (AR-7 task 1) ----->


async def test_rewrite_prompt_starts_with_the_current_date():
    model = ScriptedToolModel(
        responses=[tool_call("SubmitSearchQuery", {"query": "q"})]
    )

    await rewrite_query(model, "latest postgres news", today="2020-01-01")

    assert model.calls[0][0].startswith("Current date: 2020-01-01.")


async def test_answer_prompt_carries_the_current_date_before_context():
    model = ScriptedToolModel(
        responses=[tool_call("CitedAnswer", {"answer": "a"})]
    )

    await generate_cited_answer(model, "q", "[1] context", today="2020-01-01")

    system = model.calls[0][0]
    assert "Current date: 2020-01-01." in system
    assert system.index("Current date: 2020-01-01.") < system.index("Context Chunks:")


async def test_date_defaults_to_today_when_not_injected():
    model = ScriptedToolModel(
        responses=[tool_call("SubmitSearchQuery", {"query": "q"})]
    )

    await rewrite_query(model, "anything")

    assert model.calls[0][0].startswith("Current date: ")
