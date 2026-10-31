"""Persona CRUD and persona-scoped agent mode (integration: live Postgres)."""

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

PERSONA_PAYLOAD = {
    "name": "Rust programmer",
    "system_context": "Answers must target the Rust systems programmer.",
    "allowed_domains": ["rust-lang.org"],
    "allowed_engines": [],
    "allowed_keywords": ["rust", "cargo"],
    "default_parameters": {"max_iterations": 2},
}


@pytest_asyncio.fixture
async def auth_client(db_session):
    async def override_get_session():
        yield db_session

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_search_provider] = lambda: FakeSearchProvider()
    app.dependency_overrides[get_scraper] = lambda: make_scraper()
    app.dependency_overrides[get_ranker] = lambda: FakeRanker()
    app.dependency_overrides[get_optional_embedder] = lambda: None

    def agent_model():
        return ScriptedToolModel(
            responses=[tool_call("SubmitOutcome", {"answer": "Persona answer."})]
        )

    app.dependency_overrides[get_llm_client] = agent_model

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()

async def make_user(db_session, role_name: str | None = None) -> str:
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
        assert await assign_role_to_user(db_session, user.id, role_name)
    return create_access_token({"sub": str(user.id)})


@pytest.mark.asyncio
async def test_personas_require_authentication(auth_client):
    response = await auth_client.get("/personas")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_persona_crud_roundtrip(auth_client, db_session):
    token = await make_user(db_session, role_name="research")
    headers = {"Authorization": f"Bearer {token}"}

    created = await auth_client.post("/personas", json=PERSONA_PAYLOAD, headers=headers)
    assert created.status_code == 201, created.text
    persona = created.json()
    assert persona["name"] == "Rust programmer"
    assert persona["allowed_domains"] == ["rust-lang.org"]

    listed = await auth_client.get("/personas", headers=headers)
    assert listed.status_code == 200
    assert [p["id"] for p in listed.json()] == [persona["id"]]

    fetched = await auth_client.get(f"/personas/{persona['id']}", headers=headers)
    assert fetched.status_code == 200

    patched = await auth_client.patch(
        f"/personas/{persona['id']}",
        json={"system_context": "Updated context."},
        headers=headers,
    )
    assert patched.status_code == 200
    assert patched.json()["system_context"] == "Updated context."
    assert patched.json()["name"] == "Rust programmer"

    deleted = await auth_client.delete(f"/personas/{persona['id']}", headers=headers)
    assert deleted.status_code == 204

    gone = await auth_client.get(f"/personas/{persona['id']}", headers=headers)
    assert gone.status_code == 404


@pytest.mark.asyncio
async def test_personas_are_isolated_per_user(auth_client, db_session):
    token_a = await make_user(db_session)
    token_b = await make_user(db_session)
    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}

    created = await auth_client.post(
        "/personas", json=PERSONA_PAYLOAD, headers=headers_a
    )
    persona_id = created.json()["id"]

    # B sees nothing of A's persona: not in the list, not addressable.
    listed_b = await auth_client.get("/personas", headers=headers_b)
    assert listed_b.json() == []
    for method, call in [
        ("get", auth_client.get),
        ("patch", auth_client.patch),
        ("delete", auth_client.delete),
    ]:
        kwargs = {"json": {"name": "stolen"}} if method == "patch" else {}
        response = await call(f"/personas/{persona_id}", headers=headers_b, **kwargs)
        assert response.status_code == 404, method


@pytest.mark.asyncio
async def test_agent_with_persona_applies_it(auth_client, db_session):
    token = await make_user(db_session, role_name="research")
    headers = {"Authorization": f"Bearer {token}"}

    created = await auth_client.post("/personas", json=PERSONA_PAYLOAD, headers=headers)
    persona_id = created.json()["id"]

    # One shared model instance so the test can inspect what the route used.
    model = ScriptedToolModel(
        responses=[tool_call("SubmitOutcome", {"answer": "Persona answer."})]
    )
    app.dependency_overrides[get_llm_client] = lambda: model

    response = await auth_client.post(
        "/ask/persona",
        json={
            "query": "memory safety",
            "persona_id": persona_id,
            "depth": "quick",
        },
        headers=headers,
    )
    assert response.status_code == 200
    assert "Persona answer." in response.text

    # The persona reached the agent's system prompt (query bias).
    model = app.dependency_overrides[get_llm_client]()
    system_message = model.calls[0][0]
    assert "Rust programmer" in system_message
    assert "rust, cargo" in system_message


@pytest.mark.asyncio
async def test_agent_with_foreign_persona_is_404(auth_client, db_session):
    token_owner = await make_user(db_session)
    token_agent_user = await make_user(db_session, role_name="research")

    created = await auth_client.post(
        "/personas", json=PERSONA_PAYLOAD,
        headers={"Authorization": f"Bearer {token_owner}"},
    )
    persona_id = created.json()["id"]

    response = await auth_client.post(
        "/ask/persona",
        json={
            "query": "memory safety",
            "persona_id": persona_id,
            "depth": "quick",
        },
        headers={"Authorization": f"Bearer {token_agent_user}"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_agent_with_missing_persona_is_404(auth_client, db_session):
    token = await make_user(db_session, role_name="research")

    response = await auth_client.post(
        "/ask/persona",
        json={
            "query": "memory safety",
            "persona_id": str(uuid.uuid4()),
            "depth": "quick",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_persona_with_conversations_preserves_history(
    auth_client, db_session
):
    from uuid import UUID

    from src.modules.history.model import Conversation

    suffix = uuid.uuid4().hex[:8]
    user = await create_user(
        db_session,
        UserCreate(
            email=f"{suffix}@example.com",
            nickname=f"user_{suffix}",
            password="Sup3rSecret!pass",
        ),
    )
    token = create_access_token({"sub": str(user.id)})
    headers = {"Authorization": f"Bearer {token}"}

    created = await auth_client.post("/personas", json=PERSONA_PAYLOAD, headers=headers)
    assert created.status_code == 201
    persona_id = UUID(created.json()["id"])

    conversation = Conversation(
        user_id=user.id,
        title="persona chat",
        persona_id=persona_id,
    )
    db_session.add(conversation)
    await db_session.commit()
    conversation_id = conversation.id

    deleted = await auth_client.delete(f"/personas/{persona_id}", headers=headers)
    assert deleted.status_code == 204

    survivor = await db_session.get(Conversation, conversation_id)
    assert survivor is not None
    await db_session.refresh(survivor)
    assert survivor.persona_id is None


@pytest.mark.asyncio
async def test_init_db_fk_swap_is_idempotent(db_session):
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from src.tools.db import init_db
    from tests.conftest import _integration_database_url

    engine = create_async_engine(_integration_database_url())
    try:
        await init_db(engine=engine)
        await init_db(engine=engine)
    finally:
        await engine.dispose()
    row = await db_session.execute(
        text(
            "SELECT confdeltype FROM pg_constraint "
            "WHERE conname = 'conversation_persona_id_fkey'"
        )
    )
    confdeltype = row.scalar_one()
    if isinstance(confdeltype, bytes):
        confdeltype = confdeltype.decode()
    assert confdeltype == "n"


@pytest.mark.asyncio
async def test_init_db_flips_no_action_fk_to_set_null(db_session):
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from src.tools.db import init_db
    from tests.conftest import _integration_database_url

    # Install the pre-fix state the app DB actually has: the FK predates
    # the SET NULL change and carries a non-default constraint name, so a
    # name-based guard cannot find it.
    await db_session.execute(
        text(
            "ALTER TABLE conversation "
            "DROP CONSTRAINT conversation_persona_id_fkey"
        )
    )
    await db_session.execute(
        text(
            "ALTER TABLE conversation "
            "ADD CONSTRAINT fk_conversation_persona_id "
            "FOREIGN KEY (persona_id) REFERENCES persona(id)"
        )
    )
    await db_session.commit()
    engine = create_async_engine(_integration_database_url())
    try:
        await init_db(engine=engine)
        await init_db(engine=engine)
    finally:
        await engine.dispose()
    row = await db_session.execute(
        text(
            "SELECT c.confdeltype FROM pg_constraint c "
            "JOIN pg_attribute a ON a.attrelid = c.conrelid "
            "AND a.attnum = ANY (c.conkey) "
            "WHERE c.conrelid = 'conversation'::regclass "
            "AND c.contype = 'f' "
            "AND c.confrelid = 'persona'::regclass "
            "AND a.attname = 'persona_id'"
        )
    )
    types = [t.decode() if isinstance(t, bytes) else t for t in row.scalars().all()]
    assert types and all(t == "n" for t in types)
