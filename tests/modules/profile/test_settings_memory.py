"""Offline tests: per-model thinking/effort memory in the profile (WU-15)."""

import uuid
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from src.app import app
from src.modules.llm_provider.repository import get_repository as get_model_repo
from src.modules.profile import router as profile_router
from src.modules.profile.factory import (
    get_repository as get_prefs_repo,
)
from src.modules.profile.factory import get_settings_repository
from tests.modules.llm_provider.test_router import FakeUser
from tests.modules.profile.test_profile import (
    MODEL_ID,
    USER_ID,
    FakeModelRepo,
    FakePrefsRepo,
)

MODEL_B_ID = uuid.uuid4()
DEAD_ID = uuid.UUID("00000000-0000-0000-0000-00000000dead")


class FakeSettingsRepo:
    def __init__(self):
        self.rows: dict[tuple, tuple] = {}

    async def get(self, user_id, model_id):
        row = self.rows.get((user_id, model_id))
        if row is None:
            return None
        return SimpleNamespace(thinking=row[0], effort=row[1])

    async def upsert(self, user_id, model_id, thinking, effort):
        self.rows[(user_id, model_id)] = (thinking, effort)


@pytest.fixture
async def client():
    settings_repo = FakeSettingsRepo()
    model_repo = FakeModelRepo()
    prefs_repo = FakePrefsRepo()
    app.dependency_overrides[get_model_repo] = lambda: model_repo
    app.dependency_overrides[get_prefs_repo] = lambda: prefs_repo
    app.dependency_overrides[get_settings_repository] = lambda: settings_repo
    app.dependency_overrides[profile_router.get_current_user] = lambda: FakeUser(
        id=USER_ID, roles=[]
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, settings_repo
    app.dependency_overrides.clear()


async def put(client, body):
    response = await client.put("/profile/preferences", json=body)
    assert response.status_code == 200, response.text
    return response.json()


async def test_explicit_write_refreshes_memory(client):
    ac, settings_repo = client
    view = await put(ac, {"default_model_id": str(MODEL_ID), "thinking": True})
    assert view["resolved"]["thinking"] is True
    assert settings_repo.rows[(USER_ID, MODEL_ID)] == (True, None)


async def test_switch_back_restores_remembered(client):
    ac, _ = client
    await put(ac, {"default_model_id": str(MODEL_ID), "thinking": True})
    await put(ac, {"default_model_id": str(MODEL_B_ID)})
    view = await put(ac, {"default_model_id": str(MODEL_ID)})
    assert view["resolved"]["thinking"] is True


async def test_same_model_write_without_fields_resets(client):
    ac, settings_repo = client
    await put(ac, {"default_model_id": str(MODEL_ID), "thinking": True})
    view = await put(ac, {"default_model_id": str(MODEL_ID)})
    assert view["resolved"]["thinking"] is None
    assert settings_repo.rows[(USER_ID, MODEL_ID)] == (True, None)  # unchanged


async def test_unknown_model_cleared_without_memory_leak(client):
    ac, settings_repo = client
    await put(ac, {"default_model_id": str(MODEL_ID), "thinking": True})
    view = await put(ac, {"default_model_id": str(DEAD_ID), "thinking": False})
    assert any("Cleared default_model_id" in w for w in view["warnings"])
    assert (USER_ID, MODEL_ID) in settings_repo.rows
