"""Role gating for the elevated modes (integration: needs live Postgres)."""

import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.app import app
from src.modules.embedding.factory import get_optional_embedder
from src.modules.identity.auth.service import create_access_token
from src.modules.identity.role.service import assign_role_to_user
from src.modules.identity.user.model import UserCreate
from src.modules.identity.user.service import create_user
from src.modules.llm_provider.dependency import get_llm_client
from src.modules.owl_reranker.factory import get_ranker
from src.modules.scrapy.factory import get_scraper
from src.modules.search_engine.factory import get_search_provider
from src.tools.db import get_session
from tests.core_ask.fakes import (
    FakeRanker,
    FakeSearchProvider,
    ScriptedToolModel,
    make_scraper,
    tool_call,
)

pytestmark = pytest.mark.integration


def research_llm() -> ScriptedToolModel:
    return ScriptedToolModel(
        responses=[
            tool_call("ResearchPlan", {"subquestions": ["sq one"]}),
            tool_call(
                "ResearchReport", {"report": "## Section\nFindings. [Source](u)"}
            ),
        ]
    )


@pytest_asyncio.fixture
async def auth_client(db_session):
    async def override_get_session():
        yield db_session

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_search_provider] = lambda: FakeSearchProvider()
    app.dependency_overrides[get_scraper] = lambda: make_scraper()
    app.dependency_overrides[get_ranker] = lambda: FakeRanker()
    app.dependency_overrides[get_llm_client] = research_llm
    app.dependency_overrides[get_optional_embedder] = lambda: None

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


async def make_user_with_role(
    db_session, role_name: str | None
) -> tuple[uuid.UUID, str]:
    suffix = uuid.uuid4().hex[:8]
    user = await create_user(
        db_session,
        UserCreate(
            email=f"{suffix}@example.com",
            nickname=f"user_{suffix}",
            password="Sup3rSecret!pass",
        ),
    )
    if role_name:
        assigned = await assign_role_to_user(db_session, user.id, role_name)
        assert assigned, f"role {role_name} must be seeded"
    token = create_access_token({"sub": str(user.id)})
    return user.id, token


@pytest.mark.asyncio
async def test_research_requires_authentication(auth_client):
    response = await auth_client.get("/ask/research", params={"query": "rust"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_research_forbidden_for_plain_user_role(auth_client, db_session):
    _, token = await make_user_with_role(db_session, role_name=None)

    response = await auth_client.get(
        "/ask/research",
        params={"query": "rust", "stream": False},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403
    assert "elevated role" in response.json()["detail"]


@pytest.mark.asyncio
async def test_research_allowed_for_pro_role(auth_client, db_session):
    _, token = await make_user_with_role(db_session, role_name="pro")

    response = await auth_client.get(
        "/ask/research",
        params={"query": "rust", "stream": False},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["subquestions"] == ["sq one"]
    assert "Findings" in body["report"]


@pytest.mark.asyncio
async def test_research_allowed_for_admin_role(auth_client, db_session):
    _, token = await make_user_with_role(db_session, role_name="admin")

    response = await auth_client.get(
        "/ask/research",
        params={"query": "rust", "stream": False},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_agent_mode_forbidden_for_plain_user_role(auth_client, db_session):
    _, token = await make_user_with_role(db_session, role_name=None)

    response = await auth_client.get(
        "/ask/agent",
        params={"goal": "rust"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_agent_mode_streams_for_pro_role(auth_client, db_session):
    _, token = await make_user_with_role(db_session, role_name="pro")

    # Agent loop model: one tool round, then the outcome.
    app.dependency_overrides[get_llm_client] = lambda: ScriptedToolModel(
        responses=[
            tool_call("SubmitOutcome", {"answer": "Agent answer."}),
        ]
    )
    response = await auth_client.get(
        "/ask/agent",
        params={"goal": "rust"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert "event: message" in response.text
    assert "Agent answer." in response.text
    assert 'event: done' in response.text


@pytest.mark.asyncio
async def test_search_mode_allowed_for_plain_user_role(auth_client, db_session):
    """`search` mode is available to every authenticated user (PLAN 0.5)."""
    _, token = await make_user_with_role(db_session, role_name=None)

    app.dependency_overrides[get_llm_client] = lambda: ScriptedToolModel(
        responses=[
            tool_call("SubmitSearchQuery", {"query": "rust"}),
            tool_call("CitedAnswer", {"answer": "Answer. [Source](u)"}),
        ]
    )
    response = await auth_client.get(
        "/ask/search",
        params={"query": "rust", "stream": False},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
