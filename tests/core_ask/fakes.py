"""Shared offline fakes for ask-pipeline tests (no network, no DB, no models)."""

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

from src.modules.owl_reranker.base import IRanker
from src.modules.scrapy.providers.base import IScraper
from src.modules.search_engine.providers.base import ISearchProvider, SearchResult


class FakeSearchProvider(ISearchProvider):
    def __init__(self, results: list[SearchResult] | None = None):
        if results is None:
            results = [
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
        self.results = results
        self.last_engines: list[str] | None = None

    async def search(
        self, query: str, engines: list[str] | None = None
    ) -> list[SearchResult]:
        self.last_engines = engines
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
    def __init__(self):
        self.calls: list[tuple] = []

    def rank_web_search(self, query, search_data, top_n=None):
        self.calls.append(("web", query, len(search_data), top_n))
        ranked = [{"url": r.url, "score": 1.0} for r in search_data]
        return ranked[:top_n] if top_n else ranked

    def rank_chunks(self, query, chunks=None, top_n=None):
        assert chunks is not None
        self.calls.append(("chunks", query, len(chunks), top_n))
        return chunks[:top_n] if top_n else chunks


def sample_markdown() -> str:
    return (
        "# Unique Header One\n"
        "This is the first paragraph with plenty of meaningful text.\n\n"
        "## Unique Header Two\n"
        "This is the second paragraph discussing the subsection in detail."
    )


def make_scraper() -> FakeScraper:
    body = sample_markdown()
    return FakeScraper(
        pages={"http://example.com/a": body, "http://example.com/b": body}
    )


def tool_call(name: str, args: dict) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"call-{name}", "type": "tool_call"}
        ],
    )


class ScriptedToolModel(BaseChatModel):
    """Chat model that only speaks in tool calls, like a well-behaved LLM.

    Records every bind_tools() so tests can assert the pipeline never talks
    to the LLM without tools bound. Scripted responses are AIMessages with
    tool_calls (or exceptions). Streaming splits tool-call args into small
    fragments to exercise incremental decoding.
    """

    responses: list = Field(default_factory=list)
    calls: list = Field(default_factory=list)
    bindings: list = Field(default_factory=list)
    stream_piece_size: int = 7

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-test"

    def bind_tools(self, tools, **kwargs):
        self.bindings.append(list(tools))
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append([m.content for m in messages])
        response = self._next_response()
        return ChatResult(generations=[ChatGeneration(message=response)])

    def _next_response(self):
        if not self.responses:
            raise AssertionError("ScriptedToolModel ran out of responses")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append([m.content for m in messages])
        response = self._next_response()

        if response.tool_calls:
            tool_call = response.tool_calls[0]
            args_str = json.dumps(tool_call["args"])
            pieces = [
                args_str[i : i + self.stream_piece_size]
                for i in range(0, len(args_str), self.stream_piece_size)
            ] or [""]
            for index, piece in enumerate(pieces):
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            {
                                "name": tool_call["name"] if index == 0 else "",
                                "args": piece,
                                "id": tool_call.get("id") if index == 0 else "",
                                "index": 0,
                                "type": "tool_call_chunk",
                            }
                        ],
                    )
                )
        elif response.content:
            for i in range(0, len(response.content), self.stream_piece_size):
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=response.content[i : i + self.stream_piece_size]
                    )
                )
        # Plain empty message: nothing to stream.


def make_model(
    optimized_query: str = "rust programming language",
    answer: str = "Rust is a systems programming language. "
    "[Source](http://example.com/a)",
) -> ScriptedToolModel:
    return ScriptedToolModel(
        responses=[
            tool_call("SubmitSearchQuery", {"query": optimized_query}),
            tool_call("CitedAnswer", {"answer": answer}),
        ]
    )
