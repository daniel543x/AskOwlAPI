"""Offline tests for the persona chat-model decision table (API-NOTES #5).

`resolve_persona_model` owns the decision; `build_chat_model` is stubbed
at the module seam (it has its own suite) so no provider is touched.
"""

import uuid
from types import SimpleNamespace

import pytest

from src.core.research.services import persona_model
from src.core.research.services.persona_model import (
    ModelConfirm,
    resolve_persona_model,
)
from src.modules.jobs.gates import TurnResources, model_gate, search_gate
from src.modules.llm_provider.selection import EntryState, ResolvedModel


def persona(model_id=None):
    return SimpleNamespace(model=model_id)


def the_user():
    return SimpleNamespace(id=uuid.uuid4(), roles=[SimpleNamespace(name="research")])


def entry_state(
    *,
    kind="chat",
    is_active=True,
    required_role="user",
    capabilities=("tools",),
) -> EntryState:
    return EntryState(
        id=uuid.uuid4(),
        kind=kind,
        is_active=is_active,
        required_role=required_role,
        capabilities=list(capabilities),
    )


def resolved(entry_id=None):
    return ResolvedModel(
        kind="chat",
        model_name="m",
        provider_type="ollama",
        entry_id=entry_id,
        provider_id=uuid.uuid4(),
        max_concurrency=2,
        capabilities=["tools"],
    )


class StubRepo:
    def __init__(self, state_result, resolved_result):
        self.state_result = state_result
        self.resolved_result = resolved_result
        self.last_resolve = None

    async def entry_state(self, model_id):
        return self.state_result

    async def resolve_model(self, **kwargs):
        self.last_resolve = kwargs
        return self.resolved_result


async def run_resolve(monkeypatch, persona_, repo, allow=False):
    fallback = SimpleNamespace(name="fallback-client")
    fallback_turn = TurnResources(keys=[model_gate(uuid.uuid4(), 1)])
    built = SimpleNamespace(name="persona-client")
    seen = {}

    async def fake_build(repo_, factory_, usr, request_model_id=None, **kwargs):
        seen["request_model_id"] = request_model_id
        return built

    monkeypatch.setattr(persona_model, "build_chat_model", fake_build)
    outcome = await resolve_persona_model(
        persona_,
        allow,
        fallback_model=fallback,
        fallback_turn=fallback_turn,
        repo=repo,
        factory=None,
        user=the_user(),
        profile=None,
        thinking=None,
        effort=None,
        searching=SimpleNamespace(search_resource_id="s1", search_max_concurrency=3),
    )
    return outcome, fallback, fallback_turn, built, seen


@pytest.mark.asyncio
async def test_persona_model_none_keeps_fallback(monkeypatch):
    outcome, fallback, fallback_turn, _, seen = await run_resolve(
        monkeypatch, persona(None), StubRepo(None, resolved())
    )
    assert outcome == (fallback, fallback_turn)
    assert seen == {}


@pytest.mark.asyncio
async def test_unusable_model_without_consent_yields_confirm(monkeypatch):
    model_id = uuid.uuid4()
    advisory = resolved(entry_id=uuid.uuid4())
    repo = StubRepo(entry_state(is_active=False), advisory)
    outcome, _, _, _, _ = await run_resolve(
        monkeypatch, persona(model_id), repo, allow=False
    )
    assert isinstance(outcome, ModelConfirm)
    assert outcome.payload() == {
        "persona_model": str(model_id),
        "reason": "inactive",
        "fallback_model": str(advisory.entry_id),
    }
    assert repo.last_resolve is not None
    assert repo.last_resolve["requested_model_id"] is None


@pytest.mark.asyncio
async def test_unusable_model_with_consent_falls_back(monkeypatch):
    model_id = uuid.uuid4()
    repo = StubRepo(entry_state(is_active=False), resolved())
    outcome, fallback, fallback_turn, _, seen = await run_resolve(
        monkeypatch, persona(model_id), repo, allow=True
    )
    assert outcome == (fallback, fallback_turn)
    assert seen == {}


@pytest.mark.asyncio
async def test_usable_model_rebuilds_client_and_turn(monkeypatch):
    model_id = uuid.uuid4()
    mine = resolved(entry_id=model_id)
    repo = StubRepo(entry_state(), mine)
    outcome, fallback, _, built, seen = await run_resolve(
        monkeypatch, persona(model_id), repo
    )
    model, turn = outcome
    assert model is built
    assert seen["request_model_id"] == model_id
    assert turn.keys == [
        model_gate(mine.provider_id, 2),
        search_gate("s1", 3),
    ]
    assert turn.model_capabilities == ["tools"]
