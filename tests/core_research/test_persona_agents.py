"""Offline tests for persona-scoped agents (no network, no DB, no models)."""

from types import SimpleNamespace

import pytest

from src.core.research.services.agent import agent_logic, persona_system_block
from src.core.research.services.budgets import Budget
from src.core.research.services.pipeline import make_budget
from src.core.tools.toolbox import AskToolbox
from tests.core_ask.fakes import (
    FakeRanker,
    FakeSearchProvider,
    ScriptedToolModel,
    make_scraper,
    tool_call,
)


def rust_persona(**overrides) -> SimpleNamespace:
    persona = SimpleNamespace(
        name="Rust programmer",
        system_context="Answers must target the Rust systems programmer.",
        allowed_domains=["rust-lang.org"],
        allowed_engines=[],
        allowed_keywords=["rust", "cargo"],
        default_parameters={},
    )
    for key, value in overrides.items():
        setattr(persona, key, value)
    return persona


def mixed_search() -> FakeSearchProvider:
    from src.modules.search_engine.providers.base import SearchResult

    return FakeSearchProvider(
        results=[
            SearchResult("Rust book", "learn rust", "https://doc.rust-lang.org/book"),
            SearchResult("Peptides", "not rust", "http://example.com/peptides"),
            SearchResult("Rust blog", "rust news", "https://blog.rust-lang.org/inside"),
            SearchResult("Fake rust", "lookalike", "https://eviltrust-lang.org/fake"),
        ]
    )


def plain_toolbox(search=None) -> AskToolbox:
    return AskToolbox(search or mixed_search(), make_scraper(), FakeRanker())


async def collect(gen):
    return [event async for event in gen]


def big_budget() -> Budget:
    return Budget(max_iterations=8, time_budget_seconds=60.0, token_budget=100_000)


# <----- DoD: the persona demonstrably changes sources and answers ----->


@pytest.mark.asyncio
async def test_persona_changes_sources_for_same_query():
    """With the persona: only rust-lang.org sources; without: everything."""

    async def sources_for(toolbox):
        model = ScriptedToolModel(
            responses=[tool_call("SubmitOutcome", {"answer": "done"})]
        )
        # Topical query so the production sanity filter keeps the fixture
        # results; this test exercises domain constraints, not hygiene.
        events = await collect(
            agent_logic("rust memory safety", model, toolbox, big_budget())
        )
        assert not [e for e in events if e[0] == "error"]
        await toolbox.web_search("rust memory safety")
        return [s["url"] for s in toolbox.ranked_sources]

    constrained = AskToolbox(
        mixed_search(),
        make_scraper(),
        FakeRanker(),
        allowed_domains=rust_persona().allowed_domains,
    )
    with_persona = await sources_for(constrained)
    without_persona = await sources_for(plain_toolbox())

    assert with_persona == [
        "https://doc.rust-lang.org/book",
        "https://blog.rust-lang.org/inside",
    ]
    assert len(without_persona) == 4
    assert with_persona != without_persona


@pytest.mark.asyncio
async def test_persona_changes_the_agent_system_prompt():
    model = ScriptedToolModel(
        responses=[tool_call("SubmitOutcome", {"answer": "Rust-flavored answer."})]
    )
    events = await collect(
        agent_logic(
            "memory safety",
            model,
            plain_toolbox(),
            big_budget(),
            persona=rust_persona(),
        )
    )

    assert frames(events, "done") == [{"status": "completed"}]
    system_message = model.calls[0][0]
    assert "Rust programmer" in system_message
    assert "systems programmer" in system_message
    assert "rust, cargo" in system_message


@pytest.mark.asyncio
async def test_persona_engines_are_passed_to_the_search_provider():
    search = mixed_search()
    toolbox = AskToolbox(
        search, make_scraper(), FakeRanker(), allowed_engines=["duckduckgo"]
    )
    await toolbox.web_search("anything")

    assert search.last_engines == ["duckduckgo"]


@pytest.mark.asyncio
async def test_no_persona_leaves_the_prompt_unconstrained():
    model = ScriptedToolModel(
        responses=[tool_call("SubmitOutcome", {"answer": "generic"})]
    )
    await collect(agent_logic("q", model, plain_toolbox(), big_budget()))

    assert "persona" not in model.calls[0][0]


# <----- Prompt rendering -----


def test_persona_system_block_lists_keywords_and_domains():
    block = persona_system_block(rust_persona())

    assert "Rust programmer" in block
    assert "Bias queries toward these keywords: rust, cargo" in block
    assert "restricted to these domains: rust-lang.org" in block


def test_persona_system_block_without_constraints_is_minimal():
    block = persona_system_block(
        rust_persona(allowed_keywords=[], allowed_domains=[])
    )

    assert "keywords" not in block
    assert "domains" not in block
    assert "Rust programmer" in block


# <----- Domain filter semantics -----


@pytest.mark.asyncio
async def test_domain_filter_matches_hosts_not_substring_lookalikes():
    toolbox = AskToolbox(
        mixed_search(), make_scraper(), FakeRanker(), allowed_domains=["Rust-Lang.ORG"]
    )
    await toolbox.web_search("q")

    urls = [s["url"] for s in toolbox.ranked_sources]
    assert urls == [
        "https://doc.rust-lang.org/book",
        "https://blog.rust-lang.org/inside",
    ]


@pytest.mark.asyncio
async def test_domain_filter_emptying_results_yields_no_sources():
    toolbox = AskToolbox(
        mixed_search(), make_scraper(), FakeRanker(), allowed_domains=["nomatch.dev"]
    )
    result = await toolbox.web_search("q")

    assert toolbox.ranked_sources == []
    assert result == "[]"


# <----- Budget overrides from persona default_parameters -----


def test_make_budget_applies_persona_defaults():
    budget = make_budget(
        {"max_iterations": 3, "time_budget_seconds": 60.0, "token_budget": 500}
    )

    assert budget.iterations_left == 3
    assert budget.token_budget == 500


def test_make_budget_caps_overrides_at_settings_ceiling():
    budget = make_budget({"max_iterations": 9_999, "token_budget": 10**9})

    from src.tools.settings import settings

    assert budget.iterations_left == settings.AGENT_MAX_ITERATIONS
    assert budget.token_budget == settings.AGENT_TOKEN_BUDGET


def test_make_budget_ignores_unknown_keys_and_bad_values():
    budget = make_budget({"nonsense": 1, "max_iterations": "not-a-number-x"})

    assert budget.iterations_left >= 1  # fell back without raising


def frames(events, name):
    return [data for event, data in events if event == name]
