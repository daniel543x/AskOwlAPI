import uuid

import pytest

from src.modules.llm_provider.selection import (
    TIER_DEFAULT,
    TIER_PREFERENCE,
    TIER_REQUEST,
    CatalogView,
    EntryState,
    ModelInfo,
    SelectionError,
    has_role_access,
    resolve_model_id,
    unusable_reason,
)


def make_view(
    models: dict | None = None,
    preferences: dict | None = None,
    defaults: dict | None = None,
) -> CatalogView:
    return CatalogView(
        models=models or {},
        user_preferences=preferences or {},
        defaults=defaults or {},
    )


def entry(entry_id: uuid.UUID, kind: str = "chat", role: str = "user", active=True):
    return ModelInfo(id=entry_id, kind=kind, required_role=role, is_active=active)


@pytest.fixture
def chat_a():
    return uuid.uuid4()


@pytest.fixture
def chat_b():
    return uuid.uuid4()


def test_request_param_wins(chat_a, chat_b):
    view = make_view(
        models={chat_a: entry(chat_a), chat_b: entry(chat_b)},
        preferences={"chat": chat_b},
        defaults={"chat": chat_b},
    )

    model_id, tier = resolve_model_id("chat", chat_a, view, ["user"])
    assert model_id == chat_a
    assert tier == TIER_REQUEST


def test_user_preference_beats_global_default(chat_a, chat_b):
    view = make_view(
        models={chat_a: entry(chat_a), chat_b: entry(chat_b)},
        preferences={"chat": chat_b},
        defaults={"chat": chat_a},
    )

    model_id, tier = resolve_model_id("chat", None, view, ["user"])
    assert model_id == chat_b
    assert tier == TIER_PREFERENCE


def test_global_default_when_no_preference(chat_a):
    view = make_view(models={chat_a: entry(chat_a)}, defaults={"chat": chat_a})

    model_id, tier = resolve_model_id("chat", None, view, ["user"])
    assert model_id == chat_a
    assert tier == TIER_DEFAULT


def test_nothing_resolvable_raises_with_builtin_tier():
    with pytest.raises(SelectionError) as exc_info:
        resolve_model_id("chat", None, make_view(), ["user"])
    assert exc_info.value.tier == "builtin"


def test_role_check_blocks_request_param():
    admin_model = uuid.uuid4()
    view = make_view(models={admin_model: entry(admin_model, role="admin")})

    with pytest.raises(SelectionError, match="requires role 'admin'"):
        resolve_model_id("chat", admin_model, view, ["user"])


def test_admin_passes_role_check():
    admin_model = uuid.uuid4()
    view = make_view(
        models={admin_model: entry(admin_model, role="admin")},
        defaults={"chat": admin_model},
    )

    model_id, tier = resolve_model_id("chat", None, view, ["admin"])
    assert model_id == admin_model
    assert tier == TIER_DEFAULT


def test_inactive_model_rejected():
    model_id = uuid.uuid4()
    view = make_view(models={model_id: entry(model_id, active=False)})

    with pytest.raises(SelectionError, match="not active"):
        resolve_model_id("chat", model_id, view, ["user"])


def test_unknown_model_id_raises_key_error():
    with pytest.raises(KeyError):
        resolve_model_id("chat", uuid.uuid4(), make_view(), ["user"])


def test_kind_mismatch_rejected_on_request():
    embedding_model = uuid.uuid4()
    view = make_view(models={embedding_model: entry(embedding_model, kind="embedding")})

    with pytest.raises(SelectionError, match="not 'chat'"):
        resolve_model_id("chat", embedding_model, view, ["user"])


def test_wrong_kind_preference_is_skipped(chat_a):
    embedding_model = uuid.uuid4()
    view = make_view(
        models={
            embedding_model: entry(embedding_model, kind="embedding"),
            chat_a: entry(chat_a),
        },
        preferences={"chat": embedding_model},
        defaults={"chat": chat_a},
    )

    # Preference points at an embedding model: skip it, fall to default.
    model_id, tier = resolve_model_id("chat", None, view, ["user"])
    assert model_id == chat_a
    assert tier == TIER_DEFAULT


def test_preference_of_other_kind_ignored(chat_a):
    embedding_model = uuid.uuid4()
    view = make_view(
        models={
            chat_a: entry(chat_a),
            embedding_model: entry(embedding_model, kind="embedding"),
        },
        preferences={"embedding": embedding_model},
        defaults={"chat": chat_a},
    )

    model_id, tier = resolve_model_id("chat", None, view, ["user"])
    assert model_id == chat_a


def test_has_role_access_hierarchy():
    assert has_role_access(["user"], "user")
    assert not has_role_access(["user"], "admin")
    assert has_role_access(["pro"], "user")
    assert has_role_access(["user", "admin"], "pro")
    assert has_role_access([], "user")
    # Unknown role names now deny (F-03 least-privilege default); a user
    # carrying unknown names still passes plain entries (level defaults 0).
    assert not has_role_access(["user"], "unknown-role")


def persona_state(**overrides):
    base = dict(
        id=uuid.uuid4(),
        kind="chat",
        is_active=True,
        required_role="user",
        capabilities=["tools"],
    )
    base.update(overrides)
    return EntryState(**base)


def test_unusable_reason_tokens():
    roles = ["user"]
    assert unusable_reason(None, roles) == "missing"
    assert unusable_reason(persona_state(kind="embedding"), roles) == "missing"
    assert unusable_reason(persona_state(is_active=False), roles) == "inactive"
    assert unusable_reason(persona_state(capabilities=[]), roles) == "lacks_tools"
    assert unusable_reason(persona_state(required_role="admin"), roles) == "no_role"
    assert unusable_reason(persona_state(), roles) is None
