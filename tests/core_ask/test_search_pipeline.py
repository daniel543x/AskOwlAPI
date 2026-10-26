import json
import re

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage

from src.core.ask.services.contracts import CitedAnswer, SubmitSearchQuery
from src.core.ask.services.search import (
    NO_CONTEXT_ANSWER,
    search_json_adapter,
    search_sse_adapter,
)
from tests.core_ask.fakes import (
    FakeRanker,
    FakeScraper,
    FakeSearchProvider,
    ScriptedToolModel,
    make_model,
    make_scraper,
    tool_call,
)

SSE_FRAME_RE = re.compile(
    r"^event: (?P<event>[a-z_]+)\nid: (?P<id>\d+)\ndata: (?P<data>\{.*\})\n\n$"
)

ANSWER_TEXT = (
    "Rust is a systems programming language. [Source](http://example.com/a)"
)


def parse_frames(frames: list[str]) -> list[tuple[str, dict]]:
    parsed = []
    for frame in frames:
        match = SSE_FRAME_RE.match(frame)
        assert match, f"Malformed SSE frame: {frame!r}"
        parsed.append((match.group("event"), json.loads(match.group("data"))))
    return parsed


async def collect_sse(**kwargs) -> list[str]:
    return [
        frame
        async for frame in search_sse_adapter(
            kwargs.get("query", "what is rust?"),
            kwargs.get("searching", FakeSearchProvider()),
            kwargs.get("model", make_model(answer=ANSWER_TEXT)),
            kwargs.get("scraper", make_scraper()),
            kwargs.get("ranker", FakeRanker()),
            max_urls=kwargs.get("max_urls"),
        )
    ]


# <----- Protocol v2: framing and event sequence ----->


async def test_sse_happy_path_event_sequence(monkeypatch):
    from src.core.ask.services import search as search_module

    # Debug events are conditional on settings; pin them off so the
    # documented happy-path sequence holds regardless of the local .env.
    monkeypatch.setattr(search_module.settings, "DEBUG", False)
    events = parse_frames(await collect_sse())
    names = [name for name, _ in events]

    assert names[:8] == [
        "status",  # building query
        "query",
        "status",  # searching
        "sources",
        "status",  # downloading
        "status",  # selecting context
        "context",
        "status",  # generating answer
    ]
    # One message event per streamed answer delta, then the terminal
    # citations event (WU-9) and the terminator.
    assert set(names[8:-2]) == {"message"}
    assert len(names[8:-2]) >= 1
    assert names[-2] == "citations"
    assert names[-1] == "done"


async def test_sse_frames_have_ids_framing_and_terminator():
    frames = await collect_sse()
    assert frames

    ids = []
    for frame in frames:
        assert frame.endswith("\n\n")
        assert "\ndata: " in frame
        assert "da ta" not in frame
        match = SSE_FRAME_RE.match(frame)
        assert match is not None
        ids.append(int(match.group("id")))

    assert ids == list(range(1, len(frames) + 1))


async def test_sse_query_event_carries_optimized_query():
    events = parse_frames(await collect_sse())
    by_name = dict(events)
    assert by_name["query"]["query"] == "rust programming language"
    assert by_name["query"]["tool_call"] is True


async def test_sse_sources_event_lists_ranked_results():
    events = parse_frames(await collect_sse())

    by_name = dict(events)
    sources = by_name["sources"]["sources"]
    assert [s["url"] for s in sources] == [
        "http://example.com/a",
        "http://example.com/b",
    ]
    assert sources[0]["title"] == "Result A"
    assert "score" in sources[0]
    # WU-3: fusion score rides along on every ranked source.
    assert sources[0]["fusion_score"] is not None


async def test_sse_max_urls_limits_sources_event():
    events = parse_frames(await collect_sse(max_urls=1))

    sources = dict(events)["sources"]["sources"]
    assert len(sources) == 1
    assert sources[0]["url"] == "http://example.com/a"


# <----- Numbered citations (WU-9 / AR-5) ----->


async def test_context_is_numbered_per_chunk():
    events = parse_frames(await collect_sse())

    context = dict(events)["context"]["context"]
    assert context.startswith("[1] ")
    assert "\n\n---\n\n[2] " in context


async def test_sse_citations_event_carries_validated_refs():
    # The fixture pipeline yields 4 chunks (2 pages x 2 header sections);
    # citing all of them reaches full coverage.
    answer = "One. [1] Two. [2] Three. [3] Four. [4]"
    model = make_model(answer=answer)
    events = parse_frames(await collect_sse(model=model))

    citations = dict(events)["citations"]
    assert [c["n"] for c in citations["citations"]] == [1, 2, 3, 4]
    assert citations["citations"][0]["url"] == "http://example.com/a"
    assert citations["coverage"] == 1.0
    assert citations["unknown"] == []
    # Terminal order: citations lands right before done.
    names = [name for name, _ in events]
    assert names.index("citations") == len(names) - 2


async def test_sse_citations_event_surfaces_unknown_refs():
    model = make_model(answer="As shown in [9].")
    events = parse_frames(await collect_sse(model=model))

    citations = dict(events)["citations"]
    assert citations["unknown"] == [9]
    assert citations["coverage"] == 0.0


async def test_json_mode_returns_citation_fields():
    model = make_model(answer="Claim one. [1]")
    result = await search_json_adapter(
        "what is rust?",
        FakeSearchProvider(),
        model,
        make_scraper(),
        FakeRanker(),
    )

    assert [c["n"] for c in result["citations"]] == [1, 2, 3, 4]
    assert result["citation_coverage"] == 0.25
    assert result["unknown_citations"] == []


async def test_sse_context_and_message_events():
    events = parse_frames(await collect_sse())

    by_name = dict(events)
    assert by_name["context"]["chunks"] >= 1
    assert "first paragraph" in by_name["context"]["context"]

    message_texts = [data["text"] for name, data in events if name == "message"]
    assert "".join(message_texts) == ANSWER_TEXT
    assert by_name["done"]["status"] == "completed"


# <----- The core rule: LLM only through tool calls ----->


async def test_every_llm_call_has_tools_bound():
    model = make_model(answer=ANSWER_TEXT)
    await collect_sse(model=model)

    # Rewrite step and answer step each bound a tool contract first.
    assert len(model.calls) == 2
    assert len(model.bindings) == 2
    assert model.bindings[0] == [SubmitSearchQuery]
    assert model.bindings[1] == [CitedAnswer]


async def test_answer_step_receives_only_tool_derived_context():
    model = make_model(answer=ANSWER_TEXT)
    await collect_sse(model=model)

    answer_system_prompt = model.calls[1][0]
    assert "Context Chunks:" in answer_system_prompt
    assert "first paragraph" in answer_system_prompt
    # Only the selected context chunks, never the raw pipeline payload dict.
    assert "'context'" not in answer_system_prompt
    assert "'sources'" not in answer_system_prompt


# <----- Error and fallback paths ----->


async def test_query_falls_back_to_raw_query_on_tool_protocol_error():
    model = ScriptedToolModel(
        responses=[
            AIMessage(content="no tool call here"),
            tool_call("CitedAnswer", {"answer": ANSWER_TEXT}),
        ]
    )

    events = parse_frames(await collect_sse(model=model))
    by_name = dict(events)

    assert by_name["query"]["query"] == "what is rust?"
    assert by_name["query"]["tool_call"] is False
    assert by_name["done"]["status"] == "completed"


async def test_answer_without_tool_call_is_error_event():
    model = ScriptedToolModel(
        responses=[
            tool_call("SubmitSearchQuery", {"query": "rust"}),
            AIMessage(content="free-form answer, contract violation"),
        ]
    )

    events = parse_frames(await collect_sse(model=model))
    names = [name for name, _ in events]

    assert names[-1] == "error"
    assert "done" not in names
    assert "CitedAnswer" in events[-1][1]["error"]


async def test_no_search_results_is_error_event():
    events = parse_frames(
        await collect_sse(searching=FakeSearchProvider(results=[]))
    )

    assert events[-1] == ("error", {"error": "No search results found."})


async def test_empty_context_answer_is_deterministic_no_extra_llm_call():
    model = make_model(answer=ANSWER_TEXT)
    events = parse_frames(
        await collect_sse(model=model, scraper=FakeScraper(return_empty=True))
    )

    by_name = dict(events)
    assert by_name["context"]["chunks"] == 0
    # No-context path emits exactly one deterministic message.
    messages = [data for name, data in events if name == "message"]
    assert messages == [{"text": NO_CONTEXT_ANSWER}]
    assert by_name["done"]["status"] == "no_context"

    # Only the rewrite LLM call happened; no answer LLM call without context.
    assert len(model.calls) == 1


async def test_pipeline_exception_becomes_error_event():
    class ExplodingScraper(FakeScraper):
        async def scrape(self, data_from_search):
            raise RuntimeError("boom")

    events = parse_frames(await collect_sse(scraper=ExplodingScraper()))

    assert events[-1][0] == "error"
    assert "boom" in events[-1][1]["error"]


# <----- Debug events ----->


async def test_debug_events_emitted_when_debug_enabled(monkeypatch):
    from src.core.ask.services import search as search_module

    monkeypatch.setattr(search_module.settings, "DEBUG", True)
    try:
        events = parse_frames(await collect_sse())
        names = [name for name, _ in events]
        assert "debug" in names
        stages = [data["stage"] for name, data in events if name == "debug"]
        assert "query_optimization" in stages
        # WU-3: web_search hygiene stats ride a debug event after `sources`.
        assert "web_search" in stages
        web_debug = next(
            data
            for name, data in events
            if name == "debug" and data.get("stage") == "web_search"
        )
        assert set(web_debug) >= {
            "stage", "dropped_results", "merged_results", "max_urls",
        }
    finally:
        monkeypatch.setattr(search_module.settings, "DEBUG", False)


# <----- JSON adapter ----->


async def test_json_happy_path_returns_full_document():
    model = make_model(answer=ANSWER_TEXT)
    result = await search_json_adapter(
        "what is rust?",
        FakeSearchProvider(),
        model,
        make_scraper(),
        FakeRanker(),
    )

    assert result["query"] == "what is rust?"
    assert result["optimized_query"] == "rust programming language"
    assert result["answer"] == ANSWER_TEXT
    assert [s["url"] for s in result["sources"]] == [
        "http://example.com/a",
        "http://example.com/b",
    ]
    assert "first paragraph" in result["context"]
    assert len(model.bindings) == 2


async def test_json_error_raises_http_exception():
    with pytest.raises(HTTPException) as exc_info:
        await search_json_adapter(
            "q",
            FakeSearchProvider(results=[]),
            make_model(),
            make_scraper(),
            FakeRanker(),
        )
    assert exc_info.value.status_code == 500


async def test_json_contract_violation_is_502():
    model = ScriptedToolModel(
        responses=[
            tool_call("SubmitSearchQuery", {"query": "rust"}),
            AIMessage(content="free-form"),
        ]
    )

    with pytest.raises(HTTPException) as exc_info:
        await search_json_adapter(
            "q", FakeSearchProvider(), model, make_scraper(), FakeRanker()
        )
    assert exc_info.value.status_code == 502
    assert "tool" in exc_info.value.detail


async def test_json_no_context_returns_deterministic_answer():
    model = make_model()
    result = await search_json_adapter(
        "q",
        FakeSearchProvider(),
        model,
        FakeScraper(return_empty=True),
        FakeRanker(),
    )

    assert result["answer"] == NO_CONTEXT_ANSWER
    assert result["context"] == ""
    assert len(model.calls) == 1
