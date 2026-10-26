import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from src.app import app
from src.modules.llm_provider import router as model_router
from src.modules.llm_provider.model import (
    ModelEntryCreate,
    ModelEntryUpdate,
    ProviderCreate,
    ProviderUpdate,
)
from src.modules.llm_provider.selection import SelectionError

USER_ID = uuid.uuid4()
ADMIN_ID = uuid.uuid4()


@dataclass
class FakeUser:
    id: uuid.UUID
    roles: list = field(default_factory=list)


@dataclass
class FakeEntry:
    id: uuid.UUID
    provider_id: uuid.UUID | None = None
    kind: str = "chat"
    model_name: str = "model-x"
    display_name: str = "Model X"
    is_active: bool = True
    required_role: str = "user"
    capabilities: list = field(default_factory=list)
    vector_dims: int | None = None


@dataclass
class FakeProvider:
    id: uuid.UUID
    name: str = "prov"
    provider_type: str = "ollama"
    base_url: str | None = "http://p:1"
    api_key_encrypted: str | None = None
    is_active: bool = True
    health_status: str = "up"
    last_probe_at: str | None = None


@dataclass
class FakePreference:
    user_id: uuid.UUID
    kind: str
    selected_model_id: uuid.UUID


class FakeRepo:
    """In-memory repository implementing exactly what the routes call."""

    def __init__(self):
        self.provider = FakeProvider(id=uuid.uuid4())
        self.chat_entry = FakeEntry(id=uuid.uuid4(), provider_id=self.provider.id)
        self.pro_entry = FakeEntry(
            id=uuid.uuid4(), provider_id=self.provider.id, required_role="pro"
        )
        self.embedding_entry = FakeEntry(
            id=uuid.uuid4(),
            provider_id=self.provider.id,
            kind="embedding",
            vector_dims=1024,
        )
        self.created_providers: list[ProviderCreate] = []
        self.created_entries: list[ModelEntryCreate] = []
        self.updates: list[tuple] = []
        self.deleted: list[uuid.UUID] = []
        self.defaults: dict[str, uuid.UUID] = {}
        self.preferences: dict[str, uuid.UUID] = {}
        self.probed = False

    async def get_available_models(self, user_roles, kind=None):
        everything = [
            (self.chat_entry, self.provider),
            (self.pro_entry, self.provider),
            (self.embedding_entry, self.provider),
        ]
        out = []
        for entry, provider in everything:
            if kind and entry.kind != kind:
                continue
            if entry.required_role == "pro" and "admin" not in user_roles:
                continue
            out.append((entry, provider))
        return out

    async def get_user_preferences(self, user_id):
        return [
            FakePreference(user_id=user_id, kind=k, selected_model_id=m)
            for k, m in self.preferences.items()
        ]

    async def set_user_preference(self, user_id, model_id, user_roles):
        if model_id == self.pro_entry.id:
            raise SelectionError(
                f"Model {model_id} requires role 'pro'.", tier="permission"
            )
        if model_id == uuid.UUID("00000000-0000-0000-0000-00000000dead"):
            raise SelectionError(f"Model {model_id} not found or not active.")
        self.preferences["chat"] = model_id
        return FakePreference(user_id=user_id, kind="chat", selected_model_id=model_id)

    async def create_provider(self, data: ProviderCreate):
        if data.name == "duplicate":
            raise ValueError("Provider 'duplicate' already exists.")
        self.created_providers.append(data)
        return FakeProvider(
            id=uuid.uuid4(),
            name=data.name,
            provider_type=data.provider_type,
            base_url=data.base_url,
        )

    async def list_providers(self):
        return [self.provider]

    async def update_provider(self, provider_id, data: ProviderUpdate):
        if provider_id == uuid.UUID("00000000-0000-0000-0000-00000000dead"):
            raise ValueError(f"Provider {provider_id} not found.")
        self.updates.append(("provider", provider_id))
        return FakeProvider(id=provider_id)

    async def delete_provider(self, provider_id):
        if provider_id == uuid.UUID("00000000-0000-0000-0000-000000000001"):
            raise ValueError("Provider still has catalog entries")
        self.deleted.append(provider_id)

    async def create_model(self, data: ModelEntryCreate):
        if data.kind.value == "embedding" and not data.vector_dims:
            raise ValueError("Embedding entries must declare vector_dims.")
        if data.provider_id == uuid.UUID("00000000-0000-0000-0000-00000000dead"):
            raise ValueError("Provider not found.")
        self.created_entries.append(data)
        return FakeEntry(
            id=uuid.uuid4(),
            provider_id=data.provider_id,
            kind=data.kind.value if hasattr(data.kind, "value") else data.kind,
            model_name=data.model_name,
            capabilities=list(data.capabilities or []),
        )

    async def list_models(self, kind=None):
        entries = [self.chat_entry, self.pro_entry, self.embedding_entry]
        return [e for e in entries if not kind or e.kind == kind]

    async def update_model(self, model_id, data: ModelEntryUpdate):
        if model_id == uuid.UUID("00000000-0000-0000-0000-00000000dead"):
            raise ValueError(f"Model {model_id} not found.")
        self.updates.append(("model", model_id))
        return FakeEntry(id=model_id)

    async def delete_model(self, model_id):
        self.deleted.append(model_id)

    async def get_model(self, model_id):
        return FakeEntry(id=model_id, kind="chat", model_name="gotten")

    async def set_default(self, kind, model_id):
        if model_id == uuid.UUID("00000000-0000-0000-0000-000000000002"):
            raise ValueError(f"Model {model_id} is of kind 'chat', not '{kind}'.")
        self.defaults[kind] = model_id
        return SimpleNamespace(key=f"default_model_id:{kind}", value=str(model_id))

    async def get_defaults(self):
        return dict(self.defaults)

    async def probe_providers(self):
        self.probed = True
        return [self.provider]


def regular_user():
    return FakeUser(id=USER_ID, roles=[SimpleNamespace(name="user")])


def admin_user():
    return FakeUser(id=ADMIN_ID, roles=[SimpleNamespace(name="admin")])


@pytest.fixture
async def client():
    repo = FakeRepo()

    app.dependency_overrides[model_router.get_repository] = lambda: repo
    app.dependency_overrides[model_router.get_current_user] = regular_user
    app.dependency_overrides[model_router.get_current_admin_user] = admin_user

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, repo

    app.dependency_overrides.clear()


async def test_list_models_filters_by_role(client):
    ac, repo = client
    response = await ac.get("/models")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2  # pro entry hidden from plain user
    assert all(e["required_role"] == "user" for e in data)


async def test_list_models_filters_by_kind(client):
    ac, repo = client
    response = await ac.get("/models", params={"kind": "embedding"})

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["kind"] == "embedding"
    assert data[0]["vector_dims"] == 1024


async def test_set_preference_happy_path(client):
    ac, repo = client
    response = await ac.put(f"/models/{repo.chat_entry.id}/preference")

    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "chat"
    assert body["selected_model_id"] == str(repo.chat_entry.id)


async def test_set_preference_role_denied(client):
    ac, repo = client
    response = await ac.put(f"/models/{repo.pro_entry.id}/preference")
    assert response.status_code == 403


async def test_set_preference_missing_model(client):
    ac, _ = client
    response = await ac.put(
        "/models/00000000-0000-0000-0000-00000000dead/preference"
    )
    assert response.status_code == 404


async def test_get_preferences(client):
    ac, repo = client
    await ac.put(f"/models/{repo.chat_entry.id}/preference")
    response = await ac.get("/models/preferences")

    assert response.status_code == 200
    assert response.json()[0]["selected_model_id"] == str(repo.chat_entry.id)


async def test_create_provider(client):
    ac, repo = client
    response = await ac.post(
        "/admin/models/providers",
        json={
            "name": "OpenRouter",
            "provider_type": "openai",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-secret",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "OpenRouter"
    assert "api_key" not in body
    assert body["has_api_key"] is False  # FakeRepo stores nothing


async def test_create_provider_duplicate_conflict(client):
    ac, _ = client
    response = await ac.post(
        "/admin/models/providers",
        json={"name": "duplicate", "provider_type": "openai"},
    )
    assert response.status_code == 409


async def test_update_and_delete_provider(client):
    ac, repo = client
    provider_id = repo.provider.id

    response = await ac.patch(
        f"/admin/models/providers/{provider_id}",
        json={"base_url": "http://new:1"},
    )
    assert response.status_code == 200

    response = await ac.delete(f"/admin/models/providers/{provider_id}")
    assert response.status_code == 204

    response = await ac.delete(
        "/admin/models/providers/00000000-0000-0000-0000-000000000001"
    )
    assert response.status_code == 404


async def test_create_entry_requires_vector_dims_for_embedding(client):
    ac, _ = client
    response = await ac.post(
        "/admin/models/entries",
        json={
            "provider_id": str(uuid.uuid4()),
            "kind": "embedding",
            "model_name": "bge-m3",
        },
    )
    assert response.status_code == 400
    assert "vector_dims" in response.json()["detail"]


async def test_create_entry_happy_path(client):
    ac, _ = client
    response = await ac.post(
        "/admin/models/entries",
        json={
            "provider_id": str(uuid.uuid4()),
            "kind": "chat",
            "model_name": "gpt-4o-mini",
            "capabilities": ["tools", "streaming"],
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["capabilities"] == ["tools", "streaming"]
    assert body["kind"] == "chat"


async def test_set_default_and_list_defaults(client):
    ac, repo = client
    model_id = repo.chat_entry.id

    response = await ac.put(
        "/admin/models/defaults/chat", params={"model_id": str(model_id)}
    )
    assert response.status_code == 200
    assert response.json()["model_name"] == "gotten"

    response = await ac.get("/admin/models/defaults")
    assert response.json() == {"defaults": {"chat": str(model_id)}}


async def test_set_default_kind_mismatch(client):
    ac, _ = client
    response = await ac.put(
        "/admin/models/defaults/embedding",
        params={"model_id": "00000000-0000-0000-0000-000000000002"},
    )
    assert response.status_code == 400


async def test_models_health_probes(client):
    ac, repo = client
    response = await ac.get("/health/models")

    assert response.status_code == 200
    assert repo.probed is True
    assert response.json()[0]["health_status"] == "up"


async def test_admin_endpoints_reject_non_admin(client):
    ac, _ = client
    from fastapi import HTTPException, status

    def deny():
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="no")

    app.dependency_overrides[model_router.get_current_admin_user] = deny
    try:
        response = await ac.get("/admin/models/providers")
        assert response.status_code == 403
    finally:
        app.dependency_overrides[model_router.get_current_admin_user] = admin_user
