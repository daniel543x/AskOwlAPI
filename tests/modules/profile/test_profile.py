"""Offline tests for profile preferences (PR 10 / decision 12)."""

import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import cast

import pytest
from httpx import ASGITransport, AsyncClient

from src.app import app
from src.modules.identity.user.model import User
from src.modules.llm_provider.repository import get_repository as get_model_repo
from src.modules.llm_provider.selection import (
    TIER_DEFAULT,
    TIER_PROFILE,
    TIER_REQUEST,
    CatalogView,
    ModelInfo,
    ResolvedModel,
    resolve_model_id,
)
from src.modules.profile import router as profile_router
from src.modules.profile.factory import get_repository as get_prefs_repo
from src.modules.profile.model import UserPreferences
from src.modules.profile.service import clamp_to_model, validate_sources
from tests.modules.llm_provider.test_router import FakeUser

USER_ID = uuid.uuid4()
MODEL_ID = uuid.uuid4()
DEAD_MODEL_ID = uuid.UUID("00000000-0000-0000-0000-00000000dead")


def _as_user(user):
    return cast(User, cast(object, user))


def capable_resolved() -> ResolvedModel:
    return ResolvedModel(
        kind="chat",
        model_name="capable",
        provider_type="ollama",
        capabilities=["tools", "thinking", "effort"],
        effort_levels=["low", "high"],
    )


def plain_resolved() -> ResolvedModel:
    return ResolvedModel(
        kind="chat",
        model_name="plain",
        provider_type="ollama",
        capabilities=["tools"],
    )


class FakeModelRepo:
    """MODEL_ID resolves to a thinking/effort-capable model; unknown ids
    raise KeyError (stale reference); everything else is `tools`-only."""

    def __init__(self):
        self.calls: list[tuple] = []

    async def resolve_model(
        self,
        kind,
        user_id,
        user_roles,
        requested_model_id=None,
        **kwargs,
    ):
        self.calls.append((requested_model_id, kwargs.get("profile_model_id")))
        target = requested_model_id or kwargs.get("profile_model_id")
        if target == uuid.UUID("00000000-0000-0000-0000-00000000dead"):
            raise KeyError(target)
        if target == MODEL_ID:
            return capable_resolved()
        return plain_resolved()


@dataclass
class FakePrefsRepo:
    rows: dict = field(default_factory=dict)

    async def get(self, user_id):
        return self.rows.get(user_id)

    async def save(self, user_id, data):
        row = UserPreferences(
            user_id=user_id,
            default_model_id=data.default_model_id,
            default_mode=data.default_mode,
            thinking=data.thinking,
            effort=data.effort,
            source_types=list(data.source_types or []),
            updated_at=None,
        )
        self.rows[user_id] = row
        return row


def regular_user():
    return FakeUser(id=USER_ID, roles=[SimpleNamespace(name="user")])


@pytest.fixture
async def client():
    model_repo = FakeModelRepo()
    prefs_repo = FakePrefsRepo()

    app.dependency_overrides[get_model_repo] = lambda: model_repo
    app.dependency_overrides[get_prefs_repo] = lambda: prefs_repo
    app.dependency_overrides[profile_router.get_current_user] = regular_user

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, prefs_repo, model_repo
    app.dependency_overrides.clear()


# <----- Service (pure) ----->


def test_unknown_source_types_are_dropped_with_warning():
    kept, warnings = validate_sources(["web", "bogus", "news"])
    assert kept == ["web", "news"]
    assert warnings and "bogus" in warnings[0]


def test_clamp_resets_thinking_for_non_thinking_model():
    thinking, effort, warnings = clamp_to_model(["tools"], [], True, None, MODEL_ID)
    assert thinking is None and effort is None
    assert warnings and "thinking" in warnings[0]


def test_clamp_resets_unsupported_effort_level():
    _, effort, warnings = clamp_to_model(
        ["tools", "effort"], ["low"], None, "max", MODEL_ID
    )
    assert effort is None
    assert warnings and "max" in warnings[0]


def test_clamp_keeps_supported_values():
    thinking, effort, warnings = clamp_to_model(
        ["tools", "thinking", "effort"], ["low", "high"], False, "high", MODEL_ID
    )
    assert (thinking, effort, warnings) == (False, "high", [])


# <----- Selection: the profile tier ----->


def _view(*models: ModelInfo, preferences: dict | None = None) -> CatalogView:
    view = CatalogView(models={m.id: m for m in models})
    if preferences:
        view.user_preferences.update(preferences)
    return view


def _info(model_id: uuid.UUID, kind: str = "chat") -> ModelInfo:
    return ModelInfo(id=model_id, kind=kind, required_role="user", is_active=True)


def test_profile_tier_sits_between_request_and_preference():
    request_id, profile_id = uuid.uuid4(), uuid.uuid4()
    view = _view(_info(request_id), _info(profile_id))
    roles = ["user"]

    got, tier = resolve_model_id(
        "chat", request_id, view, roles, profile_model_id=profile_id
    )
    assert (got, tier) == (request_id, TIER_REQUEST)

    got, tier = resolve_model_id(
        "chat", None, view, roles, profile_model_id=profile_id
    )
    assert (got, tier) == (profile_id, TIER_PROFILE)


def test_stale_profile_reference_falls_through():
    preference_id = uuid.uuid4()
    view = _view(_info(preference_id), preferences={"chat": preference_id})

    got, tier = resolve_model_id(
        "chat", None, view, ["user"], profile_model_id=uuid.uuid4()
    )
    assert (got, tier) == (preference_id, "preference")

    # Without a per-kind preference the chain reaches the global default.
    view_default = _view(_info(preference_id))
    view_default.defaults["chat"] = preference_id
    got, tier = resolve_model_id(
        "chat", None, view_default, ["user"], profile_model_id=uuid.uuid4()
    )
    assert (got, tier) == (preference_id, TIER_DEFAULT)


# <----- Routes ----->


async def test_put_clamps_against_the_model_and_warns(client):
    ac, prefs_repo, _ = client

    response = await ac.put(
        "/profile/preferences",
        json={
            "default_model_id": str(MODEL_ID),
            "default_mode": "research",
            "thinking": True,
            "effort": "high",
            "source_types": ["web", "bogus"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    # The capable model supports everything: values survive the clamp.
    assert body["stored"]["thinking"] is True
    assert body["stored"]["effort"] == "high"
    assert body["stored"]["default_model_id"] == str(MODEL_ID)
    assert body["resolved"]["source_types"] == ["web"]
    assert any("bogus" in w for w in body["warnings"])
    assert prefs_repo.rows[USER_ID].default_mode == "research"


async def test_put_clears_unknown_model_with_warning(client):
    ac, _, _ = client

    response = await ac.put(
        "/profile/preferences",
        json={"default_model_id": str(DEAD_MODEL_ID)},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["stored"]["default_model_id"] is None
    assert body["resolved"]["default_model_id"] is None
    assert any("default_model_id" in w for w in body["warnings"])


async def test_get_previews_model_switch_clamps(client):
    ac, prefs_repo, _ = client
    # Stored against the capable model: thinking on, effort high.
    prefs_repo.rows[USER_ID] = UserPreferences(
        user_id=USER_ID,
        default_model_id=MODEL_ID,
        thinking=True,
        effort="high",
    )

    # No ?model_id: stored default resolves capable → values survive.
    ok = await ac.get("/profile/preferences")
    assert ok.status_code == 200
    assert ok.json()["resolved"]["thinking"] is True
    assert ok.json()["resolved"]["effort"] == "high"

    # Preview against a `tools`-only model (stale profile id degrades to
    # the default chain): thinking/effort reset, stored view untouched.
    stale_profile = uuid.uuid4()
    prefs_repo.rows[USER_ID].default_model_id = stale_profile
    preview = await ac.get("/profile/preferences")

    assert preview.status_code == 200
    body = preview.json()
    assert body["stored"]["thinking"] is True
    assert body["stored"]["effort"] == "high"
    assert body["resolved"]["thinking"] is None
    assert body["resolved"]["effort"] is None
    assert body["warnings"], "clamps must be explained"


async def test_get_empty_profile_returns_nulls(client):
    ac, _, _ = client

    response = await ac.get("/profile/preferences")

    assert response.status_code == 200
    body = response.json()
    assert body["stored"]["thinking"] is None
    assert body["resolved"]["effort"] is None
    assert body["warnings"] == []


# <----- Ask-path resolution (request > profile > model default) ----->


async def test_build_chat_model_profile_fills_unset_controls():
    from src.modules.llm_provider.dependency import build_chat_model
    from tests.modules.llm_provider.test_capability_gating import (
        RecordingFactory,
    )

    class ResolvingRepo(FakeModelRepo):
        async def resolve_model(self, kind, user_id, user_roles,
                                requested_model_id=None, **kwargs):
            return capable_resolved()

    factory = RecordingFactory()
    user = _as_user(regular_user())
    profile = UserPreferences(
        user_id=USER_ID, default_model_id=MODEL_ID, thinking=True, effort="low"
    )

    await build_chat_model(
        ResolvingRepo(), factory, user, None, profile=profile
    )

    # Request left both unset → profile values reach the adapter; the
    # effort level (in the model's levels) wins over the boolean, since an
    # effort implies thinking is enabled.
    assert factory.created[0][1] == {"reasoning": "low"}


async def test_build_chat_model_request_beats_profile_and_stays_strict():
    from fastapi import HTTPException

    from src.modules.llm_provider.dependency import build_chat_model
    from tests.modules.llm_provider.test_capability_gating import (
        RecordingFactory,
    )

    class ResolvingRepo(FakeModelRepo):
        async def resolve_model(self, kind, user_id, user_roles,
                                requested_model_id=None, **kwargs):
            return plain_resolved()

    factory = RecordingFactory()
    user = _as_user(regular_user())
    profile = UserPreferences(user_id=USER_ID, thinking=True, effort="low")

    # Explicit request values are strict even when the profile would allow.
    with pytest.raises(HTTPException) as exc_info:
        await build_chat_model(
            ResolvingRepo(),
            factory,
            user,
            None,
            thinking=True,
            profile=profile,
        )
    assert exc_info.value.status_code == 422

    # Profile-sourced values clamp silently instead.
    await build_chat_model(ResolvingRepo(), factory, user, None, profile=profile)
    assert factory.created[0][1] == {}
