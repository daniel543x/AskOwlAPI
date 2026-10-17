import json
import re
from typing import Any

import pytest
from fastapi import HTTPException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from src.core.ask.services.search import search_json_adapter, search_sse_adapter
from src.modules.owl_reranker.base import IRanker
from src.modules.scrapy.providers.base import IScraper
from src.modules.search_engine.providers.base import ISearchProvider, SearchResult

VALID_MASTER_KEY = "s2w6bWtjpHynwD8xBHyDzICIRvJ1ioxmhP39k6iBKDI="

SSE_FRAME_RE = re.compile(r"^event: (?P<event>[a-z_]+)\ndata: (?P<data>\{.*\})\n\n$")

SAMPLE_HTML = """<html><head><title>Test Page</title></head><body>
<h1>Unique Header One</h1>
<p>This is the first paragraph with plenty of meaningful text for the extractor.</p>
<h2>Unique Header Two</h2>
<p>This is the second paragraph discussing the subsection in enough detail here.</p>
</body></html>"""


# <----- Fakes (no network, no DB, no real models) ----->


class FakeSearchProvider(ISearchProvider):
    def __init__(self, results: list[SearchResult] | None = None):
        self.results = results if results is not None else [
            SearchResult(
                title="Result A",
                snippet="Snippet about rust language",
                url="http://example.com/a",
            ),
            SearchResult(
                title="Result B",
                snippet="Another snippet",
                url="http://example.com/b",
            ),
        ]

    async def search(self, query: str) -> list[SearchResult]:
        return self.results


class FakeScraper(IScraper):
    def __init__(self, pages: dict | None = None, return_empty: bool = False):
        self.pages = pages or {}
        self.return_empty = return_empty
        self.requested: list[str] = []

    @property
    def name(self) -> str:
        return "fake"

    async def scrape(self, data_from_search) -> list[dict] | None:
        if self.return_empty:
            return []
        content = []
        for item in data_from_search:
            self.requested.append(item["url"])
            page = {"content": self.pages.get(item["url"], ""), "url": item["url"]}
            content.append(page)
        return content


class FakeRanker(IRanker):
    def rank_web_search(self, query, search_data, top_n=None):
        ranked = [{"url": r.url, "score": 1.0} for r in search_data]
        return ranked[:top_n] if top_n else ranked

    def rank_chunks(self, query, chunks=None, top_n=None):
        assert chunks is not None
        return chunks[:top_n] if top_n else chunks


class ScriptedChatModel(BaseChatModel):
    """Chat model returning scripted responses and recording every prompt."""

    responses: list[str] = Field(default_factory=list)
    calls: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-test"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append([m.content for m in messages])
        content = self.responses.pop(0) if self.responses else "default-test-response"
        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])


def make_model() -> ScriptedChatModel:
    return ScriptedChatModel(
        responses=[
            '{"search_query": "rust programming language"}',
            "Rust is a systems programming language. [Source](http://example.com/a)",
        ]
    )


def make_scraper() -> FakeScraper:
    body = (
        "# Unique Header One\n"
        "This is the first paragraph with enough meaningful text for extraction.\n\n"
        "## Unique Header Two\n"
        "This is the second paragraph discussing the subsection in detail."
    )
    return FakeScraper(
        pages={"http://example.com/a": body, "http://example.com/b": body}
    )


def parse_frames(frames: list[str]):
    parsed = []
    for frame in frames:
        match = SSE_FRAME_RE.match(frame)
        assert match, f"Malformed SSE frame: {frame!r}"
        parsed.append((match.group("event"), json.loads(match.group("data"))))
    return parsed


# <----- SSE adapter ----->


async def test_sse_happy_path_event_sequence_and_framing():
    events = parse_frames(
        [
            frame
            async for frame in search_sse_adapter(
                "what is rust?",
                FakeSearchProvider(),
                make_model(),
                make_scraper(),
                FakeRanker(),
            )
        ]
    )

    names = [name for name, _ in events]
    assert names[0] == "status"
    assert names[-1] == "done"
    assert names.count("message") == 1

    by_name = dict(events)
    assert by_name["query_generated"]["query"] == "rust programming language"
    assert by_name["message"]["text"].startswith("Rust is a systems")
    assert by_name["done"] == {"status": "Completed"}


async def test_sse_frames_have_correct_framing_and_terminator():
    frames = [
        frame
        async for frame in search_sse_adapter(
            "q", FakeSearchProvider(), make_model(), make_scraper(), FakeRanker()
        )
    ]
    assert frames, "expected at least one frame"
    for frame in frames:
        assert frame.endswith("\n\n")
        assert "da ta" not in frame
        assert "\ndata: " in frame


async def test_sse_error_event_when_no_search_results():
    events = parse_frames(
        [
            frame
            async for frame in search_sse_adapter(
                "q",
                FakeSearchProvider(results=[]),
                make_model(),
                make_scraper(),
                FakeRanker(),
            )
        ]
    )

    assert events[-1][0] == "error"
    assert events[-1][1]["error"] == "No search results found."


async def test_sse_warning_event_when_nothing_scraped():
    events = parse_frames(
        [
            frame
            async for frame in search_sse_adapter(
                "q",
                FakeSearchProvider(),
                make_model(),
                FakeScraper(return_empty=True),
                FakeRanker(),
            )
        ]
    )

    assert events[-1][0] == "warning"
    assert "No page content" in events[-1][1]["warning"]


async def test_sse_critical_pipeline_error_is_error_event_not_exception():
    class ExplodingScraper(FakeScraper):
        async def scrape(self, data_from_search):
            raise RuntimeError("boom")

    events = parse_frames(
        [
            frame
            async for frame in search_sse_adapter(
                "q",
                FakeSearchProvider(),
                make_model(),
                ExplodingScraper(),
                FakeRanker(),
            )
        ]
    )

    assert events[-1][0] == "error"
    assert "boom" in events[-1][1]["error"]


async def test_sse_answer_generation_failure_is_error_event():
    class ExplodingAfterContextModel(ScriptedChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.calls.append([m.content for m in messages])
            if self.responses:
                content = self.responses.pop(0)
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=content))]
                )
            raise RuntimeError("llm exploded during streaming")

    events = parse_frames(
        [
            frame
            async for frame in search_sse_adapter(
                "q",
                FakeSearchProvider(),
                ExplodingAfterContextModel(
                    responses=['{"search_query": "rust programming language"}']
                ),
                make_scraper(),
                FakeRanker(),
            )
        ]
    )

    assert events[-1][0] == "error"
    assert "Answer generation failed" in events[-1][1]["error"]
    assert "done" not in [name for name, _ in events]


# <----- JSON adapter ----->


async def test_json_happy_path_returns_query_answer_sources():
    result = await search_json_adapter(
        "what is rust?",
        FakeSearchProvider(),
        make_model(),
        make_scraper(),
        FakeRanker(),
    )

    assert result["query"] == "what is rust?"
    assert result["answer"].startswith("Rust is a systems")
    assert result["sources"] == FakeSearchProvider().results


async def test_json_adapter_raises_http_exception_on_error():
    with pytest.raises(HTTPException) as exc_info:
        await search_json_adapter(
            "q",
            FakeSearchProvider(results=[]),
            make_model(),
            make_scraper(),
            FakeRanker(),
        )
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "No search results found."


# <----- Context wiring ----->


async def test_answer_step_receives_context_string_not_payload_dict():
    model = make_model()
    await search_json_adapter(
        "what is rust?", FakeSearchProvider(), model, make_scraper(), FakeRanker()
    )

    assert len(model.calls) == 2
    answer_system_prompt = model.calls[1][0]
    # The context chunks (not the whole {context, sources, query} payload dict).
    assert "Context Chunks:" in answer_system_prompt
    assert "enough meaningful text" in answer_system_prompt
    assert "'context'" not in answer_system_prompt
    assert "'sources'" not in answer_system_prompt
    assert "User Query: what is rust?" in model.calls[1][1]


async def test_query_falls_back_to_raw_query_on_unparseable_llm_output():
    model = ScriptedChatModel(responses=["this is not json at all"])

    events = parse_frames(
        [
            frame
            async for frame in search_sse_adapter(
                "original question",
                FakeSearchProvider(),
                model,
                make_scraper(),
                FakeRanker(),
            )
        ]
    )

    by_name = dict(events)
    assert by_name["query_generated"]["query"] == "original question"


async def test_top_n_limits_flow_from_settings():
    scraper = make_scraper()
    await search_json_adapter(
        "q", FakeSearchProvider(), make_model(), scraper, FakeRanker()
    )

    from src.tools.settings import settings

    assert len(scraper.requested) <= settings.TOP_N_URLS
