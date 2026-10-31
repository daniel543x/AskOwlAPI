"""Offline tests: numbered citations across the search answer paths."""

import json
import re

from src.core.ask.services.search import (
    search_json_adapter,
    search_sse_adapter,
)
from tests.core_ask.fakes import (
    FakeRanker,
    FakeScraper,
    FakeSearchProvider,
    make_model,
)

SSE_FRAME_RE = re.compile(
    r"^event: (?P<event>[a-z_]+)\nid: (?P<id>\d+)\ndata: (?P<data>\{.*\})\n\n$"
)

FIVE = [f"https://example.com/s{i}" for i in range(1, 6)]


def _five_provider():
    from src.modules.search_engine.providers.base import SearchResult

    return FakeSearchProvider(
        results=[
            SearchResult(
                f"Rust result {i}", f"rust snippet {i}", url
            )
            for i, url in enumerate(FIVE, 1)
        ]
    )


def _five_scraper():
    return FakeScraper(pages={url: f"page {url}" for url in FIVE})


def parse_frames(frames):
    parsed = []
    for frame in frames:
        match = SSE_FRAME_RE.match(frame)
        assert match, f"Malformed SSE frame: {frame!r}"
        parsed.append((match.group("event"), json.loads(match.group("data"))))
    return parsed


async def _run_sse(results_provider, model):
    frames = [
        frame
        async for frame in search_sse_adapter(
            "what is rust?",
            results_provider,
            model,
            _five_scraper(),
            FakeRanker(),
        )
    ]
    return parse_frames(frames)


async def test_sse_numbers_dense_by_first_citation():
    model = make_model(
        answer=f"Late [Source]({FIVE[3]}) then [Source]({FIVE[0]})."
    )
    events = await _run_sse(_five_provider(), model)

    message_text = "".join(data["text"] for name, data in events if name == "message")
    assert "[1]" in message_text and "[2]" in message_text
    assert "[Source](" not in message_text

    names = [name for name, _ in events]
    citations_index = names.index("citations")
    assert names[citations_index + 1] == "done"
    mapping = {
        item["n"]: item["url"] for item in events[citations_index][1]["citations"]
    }
    assert mapping == {1: FIVE[3], 2: FIVE[0], 3: FIVE[1], 4: FIVE[2], 5: FIVE[4]}


async def test_sse_split_link_never_leaks():
    model = make_model(
        answer="Split [Sou" + f"rce]({FIVE[0]}) tail [Source]({FIVE[1]})."
    )
    events = await _run_sse(_five_provider(), model)

    for name, data in events:
        if name == "message":
            assert "[Source](" not in data["text"]
    message_text = "".join(data["text"] for name, data in events if name == "message")
    assert "[1]" in message_text and "[2]" in message_text


async def test_json_sources_match_sse_numbering():
    model = make_model(
        answer=f"Late [Source]({FIVE[3]}) then [Source]({FIVE[0]})."
    )
    provider = _five_provider()

    result = await search_json_adapter(
        "what is rust?", provider, model, _five_scraper(), FakeRanker()
    )
    assert result["citations"] == [
        {"n": 1, "url": FIVE[3]},
        {"n": 2, "url": FIVE[0]},
        {"n": 3, "url": FIVE[1]},
        {"n": 4, "url": FIVE[2]},
        {"n": 5, "url": FIVE[4]},
    ]
    assert "Late [1] then [2]." == result["answer"]


async def test_no_sources_answer_is_unchanged(monkeypatch):
    from src.core.ask.services import search as search_module

    monkeypatch.setattr(search_module.settings, "DEBUG", False)
    empty = FakeSearchProvider(results=[])
    events = await _run_sse(empty, make_model())

    names = [name for name, _ in events]
    assert "citations" not in names
    assert names[-1] == "done" or "error" in names
