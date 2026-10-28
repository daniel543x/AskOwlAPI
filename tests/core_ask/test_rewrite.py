"""Offline tests for history-aware standalone rewriting (PR 4 / WU-1)."""


from src.core.ask.services.llm_steps import rewrite_query
from tests.core_ask.fakes import (
    FakeRanker,
    FakeScraper,
    FakeSearchProvider,
    ScriptedToolModel,
    tool_call,
)

# <----- rewrite_query (history-aware) ----->


async def test_history_prepends_prior_turns_and_sets_standalone():
    model = ScriptedToolModel(
        responses=[tool_call("SubmitSearchQuery", {"query": "rust async"})]
    )
    history = [("user", "tell me about rust"), ("assistant", "Rust is a language.")]

    optimized, used_tool_call, standalone = await rewrite_query(
        model, "what about its speed?", history
    )

    assert optimized == "rust async"
    assert used_tool_call is True and standalone is True
    prompt = model.calls[0][-1]
    assert "User: tell me about rust" in prompt
    assert "Current question: what about its speed?" in prompt


async def test_no_history_keeps_standalone_false():
    model = ScriptedToolModel(
        responses=[tool_call("SubmitSearchQuery", {"query": "rust"})]
    )
    _, _, standalone = await rewrite_query(model, "rust", None)
    assert standalone is False


async def test_simple_mode_passes_history_free(monkeypatch):
    model = ScriptedToolModel(
        responses=[tool_call("SubmitSearchQuery", {"query": "raw"})]
    )
    history = [("user", "old turn")]

    _, _, standalone = await rewrite_query(model, "raw", history)

    # simple mode: history is threaded by search_logic, so the unit only
    # reflects what it was given — with history it is standalone.
    assert standalone is True
    assert "old turn" in model.calls[0][-1]


# <----- Pipeline: the rewrite modes ----->


def _pipeline_model():
    return ScriptedToolModel(
        responses=[
            tool_call("SubmitSearchQuery", {"query": "rust"}),
            tool_call("CitedAnswer", {"answer": "Answer. [Source](u)"}),
        ]
    )


async def _run_pipeline(model, **kwargs):
    from src.core.ask.services.search import search_logic

    events = []
    async for event, data in search_logic(
        kwargs.pop("query", "and what about its speed?"),
        kwargs.pop("searching", FakeSearchProvider()),
        model,
        kwargs.pop("scraper", FakeScraper(
            pages={"http://example.com/a": "Rust ownership."}
        )),
        kwargs.pop("ranker", FakeRanker()),
        **kwargs,
    ):
        events.append((event, data))
    return events


async def test_default_is_standalone_with_history():
    model = _pipeline_model()
    history = [("user", "tell me about rust"), ("assistant", "Rust is a language.")]
    events = await _run_pipeline(model, history=history)

    query_event = next(d for e, d in events if e == "query")
    assert query_event["standalone"] is True
    # The rewrite prompt carried the prior turns.
    assert "User: tell me about rust" in model.calls[0][-1]


async def test_simple_mode_never_sees_history():
    model = _pipeline_model()
    history = [("user", "tell me about rust")]
    events = await _run_pipeline(
        model, history=history, rewrite="simple"
    )

    query_event = next(d for e, d in events if e == "query")
    assert query_event["standalone"] is False
    assert "tell me about rust" not in model.calls[0][-1]


async def test_none_skips_the_rewriting_llm_call():
    model = ScriptedToolModel(
        responses=[tool_call("CitedAnswer", {"answer": "Answer. [Source](u)"})]
    )
    events = await _run_pipeline(
        model, query="rust", rewrite="none",
        searching=FakeSearchProvider(),
        scraper=FakeScraper(pages={"http://example.com/a": "Rust ownership."}),
        ranker=FakeRanker(),
    )

    query_event = next(d for e, d in events if e == "query")
    assert query_event["tool_call"] is False
    assert query_event["standalone"] is False
    assert all("SubmitSearchQuery" not in str(b) for b in model.bindings)


def _make_pipeline_toolbox():
    from src.core.tools.toolbox import AskToolbox

    pages = {
        "http://example.com/a": "# Rust ownership\\nOwnership makes rust safe."
    }
    return AskToolbox(
        FakeSearchProvider(), FakeScraper(pages=pages), FakeRanker(), None, None
    )
