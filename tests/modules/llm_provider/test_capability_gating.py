import uuid

import pytest
from fastapi import HTTPException

from src.modules.llm_provider.dependency import (
    MissingCapabilityError,
    ensure_capability,
    get_llm_client,
)
from src.modules.llm_provider.factory import ChatModelFactory
from src.modules.llm_provider.selection import ResolvedModel
from tests.modules.llm_provider.test_router import FakeUser


def resolved_with_capabilities(capabilities: list[str]) -> ResolvedModel:
    return ResolvedModel(
        kind="chat",
        model_name="some-model",
        provider_type="ollama",
        base_url="http://ollama:11434",
        capabilities=capabilities,
    )


def test_ensure_capability_passes_when_present():
    ensure_capability(resolved_with_capabilities(["tools", "streaming"]), "tools")


def test_ensure_capability_raises_when_missing():
    with pytest.raises(MissingCapabilityError, match="'tools' capability"):
        ensure_capability(resolved_with_capabilities(["streaming"]), "tools")


class FakeRepo:
    def __init__(self, resolved: ResolvedModel):
        self.resolved = resolved
        self.resolved_with = None

    async def resolve_model(self, kind, user_id, user_roles, requested_model_id=None):
        self.resolved_with = (kind, requested_model_id)
        return self.resolved


class RecordingFactory(ChatModelFactory):
    def __init__(self):
        self.created = []
        self.sentinel = object()

    def create_chat_model(self, resolved):
        self.created.append(resolved)
        return self.sentinel


async def test_get_llm_client_gates_on_tools_capability():
    repo = FakeRepo(resolved_with_capabilities([]))
    factory = RecordingFactory()
    user = FakeUser(id=uuid.uuid4(), roles=[])

    with pytest.raises(HTTPException) as exc_info:
        await get_llm_client(repo=repo, factory=factory, user=user)

    assert exc_info.value.status_code == 422
    assert "tools" in exc_info.value.detail
    assert factory.created == []


async def test_get_llm_client_builds_model_when_capable():
    repo = FakeRepo(resolved_with_capabilities(["tools"]))
    factory = RecordingFactory()
    user = FakeUser(id=uuid.uuid4(), roles=[])

    model = await get_llm_client(
        repo=repo, factory=factory, user=user, request_model_id=None
    )

    assert model is factory.sentinel
    assert repo.resolved_with[0] == "chat"
