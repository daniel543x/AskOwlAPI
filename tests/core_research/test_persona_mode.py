"""Offline tests for persona mode (PR 20 / decision 10)."""

import uuid

from src.core.research.services.budgets import Budget
from src.core.research.services.pipeline import research_logic
from src.modules.personas.model import Persona
from src.tools.settings import settings
from tests.core_ask.fakes import (
    FakeRanker,
    FakeScraper,
    FakeSearchProvider,
    ScriptedToolModel,
    tool_call,
)


def _persona(**default_parameters) -> Persona:
    return Persona(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        name="rust expert",
        system_context="Focus on rust.",
        allowed_domains=["example.com"],
        default_parameters=default_parameters,
    )


def _make_toolbox(persona=None):
    from src.core.tools.toolbox import AskToolbox

    pages = {
        "http://example.com/a": "# Rust ownership\\nOwnership makes rust safe.",
    }
    return AskToolbox(
        FakeSearchProvider(),
        FakeScraper(pages=pages),
        FakeRanker(),
        None,
        None,
        allowed_domains=persona.allowed_domains if persona else None,
        allowed_engines=persona.allowed_engines if persona else None,
    )


async def _collect(gen):
    events = []
    async for event, data in gen:
        events.append((event, data))
    return events


# <----- Depth resolution ----->


def test_depth_request_beats_persona_and_default():
    from src.core.research.router import PersonaAskRequest, _resolve_depth

    persona = _persona(depth="deep")
    request = PersonaAskRequest(
        query="q", persona_id=persona.id, depth="quick"
    )
    assert _resolve_depth(request, persona) == "quick"

    request = PersonaAskRequest(query="q", persona_id=persona.id)
    assert _resolve_depth(request, persona) == "deep"

    empty = _persona()
    request = PersonaAskRequest(query="q", persona_id=empty.id)
    assert _resolve_depth(request, empty) == settings.PERSONA_DEFAULT_DEPTH


def test_invalid_persona_depth_falls_back():
    from src.core.research.router import PersonaAskRequest, _resolve_depth

    persona = _persona(depth="turbo")
    request = PersonaAskRequest(query="q", persona_id=persona.id)
    assert _resolve_depth(request, persona) == settings.PERSONA_DEFAULT_DEPTH


# <----- Persona constraints reach the pipeline toolbox ----->


async def test_persona_constraints_scope_the_research_toolbox():
    persona = _persona()
    captured = {}

    class RecordingToolbox:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        # pipeline methods used before synthesis are no-ops here
        async def web_search(self, query):
            return "{}"

        async def fetch_pages(self):
            return ""

        async def get_context(self, query):
            return ""

    import src.core.research.services.pipeline as pipeline_mod

    original = pipeline_mod.AskToolbox
    pipeline_mod.AskToolbox = RecordingToolbox
    try:
        model = ScriptedToolModel(
            responses=[
                tool_call("ResearchPlan", {"subquestions": []}),
            ]
        )
        await _collect(
            research_logic(
                "rust",
                FakeSearchProvider(),
                model,
                FakeScraper(),
                FakeRanker(),
                persona=persona,
            )
        )
    finally:
        pipeline_mod.AskToolbox = original

    assert captured["allowed_domains"] == ["example.com"]


async def test_persona_constraints_do_not_leak_without_persona():
    captured = {}

    class RecordingToolbox:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        async def web_search(self, query):
            return "{}"

        async def fetch_pages(self):
            return ""

        async def get_context(self, query):
            return ""

    import src.core.research.services.pipeline as pipeline_mod

    original = pipeline_mod.AskToolbox
    pipeline_mod.AskToolbox = RecordingToolbox
    try:
        model = ScriptedToolModel(
            responses=[tool_call("ResearchPlan", {"subquestions": []})]
        )
        await _collect(
            research_logic(
                "rust", FakeSearchProvider(), model, FakeScraper(), FakeRanker()
            )
        )
    finally:
        pipeline_mod.AskToolbox = original

    assert captured["allowed_domains"] is None


# <----- Persona budget clamping (pre-existing contract) ----->


def test_persona_budget_clamped_by_bounded():
    from src.core.research.services.pipeline import make_budget

    persona = _persona(max_iterations=99, time_budget_seconds=1, token_budget=5)
    budget = make_budget(persona.default_parameters)
    assert isinstance(budget, Budget)
    assert budget.max_iterations <= settings.AGENT_MAX_ITERATIONS
    assert budget.token_budget <= settings.AGENT_TOKEN_BUDGET
