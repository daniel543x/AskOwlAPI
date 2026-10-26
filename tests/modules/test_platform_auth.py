"""Platform hardening: admin users, API keys, history (integration)."""

import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.app import app
from src.modules.embedding.factory import get_optional_embedder
from src.modules.identity.auth import security as auth_security
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


def search_model(answer="Cited answer. [Source](http://example.com/a)"):
    return ScriptedToolModel(
        responses=[
            tool_call("SubmitSearchQuery", {"query": "rust"}),
            tool_call("CitedAnswer", {"answer": answer}),
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
    app.dependency_overrides[get_optional_embedder] = lambda: None
    app.dependency_overrides[get_llm_client] = search_model

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


async def make_user(db_session, role_name: str | None = None):
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
    token = create_access_token({"sub": str(user.id)})
    return user, token


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# <----- Admin user management ----->


@pytest.mark.asyncio
async def test_admin_lists_users(auth_client, db_session):
    admin, admin_token = await make_user(db_session, role_name="admin")
    _, plain_token = await make_user(db_session)

    as_admin = await auth_client.get("/admin/users", headers=bearer(admin_token))
    assert as_admin.status_code == 200
    assert any(u["id"] == str(admin.id) for u in as_admin.json())

    as_plain = await auth_client.get("/admin/users", headers=bearer(plain_token))
    assert as_plain.status_code == 403


@pytest.mark.asyncio
async def test_blocking_a_user_kills_their_access(auth_client, db_session):
    admin, admin_token = await make_user(db_session, role_name="admin")
    victim, victim_token = await make_user(db_session)

    me_before = await auth_client.get("/auth/users/me", headers=bearer(victim_token))
    assert me_before.status_code == 200

    blocked = await auth_client.patch(
        f"/admin/users/{victim.id}/status",
        json={"is_active": False},
        headers=bearer(admin_token),
    )
    assert blocked.status_code == 200
    assert blocked.json()["is_active"] is False

    me_after = await auth_client.get("/auth/users/me", headers=bearer(victim_token))
    assert me_after.status_code == 403
    assert "disabled" in me_after.json()["detail"]

    login = await auth_client.post(
        "/auth/login",
        json={
            "username_or_email": victim.email,
            "password": "Sup3rSecret!pass",
        },
    )
    assert login.status_code == 401

    unblocked = await auth_client.patch(
        f"/admin/users/{victim.id}/status",
        json={"is_active": True},
        headers=bearer(admin_token),
    )
    assert unblocked.status_code == 200
    assert (
        await auth_client.get("/auth/users/me", headers=bearer(victim_token))
    ).status_code == 200


@pytest.mark.asyncio
async def test_admin_cannot_block_self(auth_client, db_session):
    admin, admin_token = await make_user(db_session, role_name="admin")

    response = await auth_client.patch(
        f"/admin/users/{admin.id}/status",
        json={"is_active": False},
        headers=bearer(admin_token),
    )
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_blocked_users_api_key_is_rejected(auth_client, db_session):
    admin, admin_token = await make_user(db_session, role_name="admin")
    victim, _ = await make_user(db_session)

    created = await auth_client.post(
        "/api-keys",
        json={"name": "victim key", "scopes": ["search"]},
        headers=bearer(create_access_token({"sub": str(victim.id)})),
    )
    assert created.status_code == 201
    raw_key = created.json()["raw_key"]

    ok_before = await auth_client.post(
        "/ask/search",
        json={"query": "rust", "stream": False},
        headers={"X-API-Key": raw_key},
    )
    assert ok_before.status_code == 200

    await auth_client.patch(
        f"/admin/users/{victim.id}/status",
        json={"is_active": False},
        headers=bearer(admin_token),
    )
    rejected = await auth_client.post(
        "/ask/search",
        json={"query": "rust", "stream": False},
        headers={"X-API-Key": raw_key},
    )
    assert rejected.status_code == 401


# <----- API keys ----->


@pytest.mark.asyncio
async def test_api_key_lifecycle(auth_client, db_session):
    user, token = await make_user(db_session)

    created = await auth_client.post(
        "/api-keys",
        json={"name": "my key", "scopes": ["search", "research"]},
        headers=bearer(token),
    )
    assert created.status_code == 201
    body = created.json()
    assert body["raw_key"].startswith("ak-")
    assert body["scopes"] == ["search", "research"]
    assert "key_hash" not in body

    listed = await auth_client.get("/api-keys", headers=bearer(token))
    assert listed.status_code == 200
    assert [k["name"] for k in listed.json()] == ["my key"]
    assert all("raw_key" not in k and "key_hash" not in k for k in listed.json())
    assert listed.json()[0]["last_used_at"] is None

    used = await auth_client.post(
        "/ask/search",
        json={"query": "rust", "stream": False},
        headers={"X-API-Key": body["raw_key"]},
    )
    assert used.status_code == 200

    listed_after = await auth_client.get("/api-keys", headers=bearer(token))
    assert listed_after.json()[0]["last_used_at"] is not None

    revoked = await auth_client.delete(
        f"/api-keys/{body['id']}", headers=bearer(token)
    )
    assert revoked.status_code == 204
    after_revoke = await auth_client.post(
        "/ask/search",
        json={"query": "rust", "stream": False},
        headers={"X-API-Key": body["raw_key"]},
    )
    assert after_revoke.status_code == 401


@pytest.mark.asyncio
async def test_api_key_unknown_scope_is_rejected(auth_client, db_session):
    _, token = await make_user(db_session)

    response = await auth_client.post(
        "/api-keys", json={"name": "bad", "scopes": ["root"]}, headers=bearer(token)
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_api_key_scopes_gate_modes(auth_client, db_session):
    _, token = await make_user(db_session, role_name="research")

    created = await auth_client.post(
        "/api-keys",
        json={"name": "narrow", "scopes": ["search"]},
        headers=bearer(token),
    )
    raw_key = created.json()["raw_key"]

    search_ok = await auth_client.post(
        "/ask/search",
        json={"query": "rust", "stream": False},
        headers={"X-API-Key": raw_key},
    )
    assert search_ok.status_code == 200

    research_denied = await auth_client.get(
        "/ask/research", params={"query": "rust", "stream": False},
        headers={"X-API-Key": raw_key},
    )
    assert research_denied.status_code == 403
    assert "'research' scope" in research_denied.json()["detail"]


@pytest.mark.asyncio
async def test_api_key_rate_limit(auth_client, db_session):
    _, token = await make_user(db_session)
    created = await auth_client.post(
        "/api-keys",
        json={"name": "limited", "scopes": ["search"]},
        headers=bearer(token),
    )
    raw_key = created.json()["raw_key"]

    original_limit = auth_security._rate_limiter.limit
    auth_security._rate_limiter.limit = 2
    auth_security._rate_limiter._windows.clear()
    try:
        statuses = [
            (
                await auth_client.post(
        "/ask/search",
        json={"query": "rust", "stream": False},
        headers={"X-API-Key": raw_key},
    )
            ).status_code
            for _ in range(3)
        ]
    finally:
        auth_security._rate_limiter.limit = original_limit
        auth_security._rate_limiter._windows.clear()

    assert statuses == [200, 200, 429]


@pytest.mark.asyncio
async def test_bogus_api_key_is_401(auth_client, db_session):
    await make_user(db_session)  # at least one user exists

    response = await auth_client.post(
        "/ask/search",
        json={"query": "rust", "stream": False},
        headers={"X-API-Key": "ak-does-not-exist"},
    )
    assert response.status_code == 401


# <----- History ----->


@pytest.mark.asyncio
async def test_search_persists_and_reuses_conversations(auth_client, db_session):
    user, token = await make_user(db_session)

    first = await auth_client.post(
        "/ask/search",
        json={"query": "first question", "stream": False},
        headers=bearer(token,
    ),
    )
    assert first.status_code == 200
    conversation_id = first.json()["answer"]["conversation_id"]
    assert conversation_id

    listed = await auth_client.get("/history", headers=bearer(token))
    assert listed.status_code == 200
    assert [c["id"] for c in listed.json()] == [conversation_id]
    assert listed.json()[0]["title"] == "first question"

    fetched = await auth_client.get(
        f"/history/{conversation_id}", headers=bearer(token)
    )
    assert fetched.status_code == 200
    roles = [m["role"] for m in fetched.json()["messages"]]
    assert roles == ["user", "assistant"]

    # Second turn in the same conversation: history must reach the LLM.
    model = search_model(answer="Second answer. [Source](http://example.com/a)")
    app.dependency_overrides[get_llm_client] = lambda: model
    second = await auth_client.post(
        "/ask/search",
        json={"query": "second question", "stream": False,
                "conversation_id": conversation_id},
        headers=bearer(token,
    ),
    )
    assert second.status_code == 200
    assert second.json()["answer"]["conversation_id"] == conversation_id

    # The answer step received the prior turn as chat history.
    answer_messages = model.calls[-1]
    assert "Cited answer." in str(answer_messages)

    fetched_again = await auth_client.get(
        f"/history/{conversation_id}", headers=bearer(token)
    )
    assert [m["role"] for m in fetched_again.json()["messages"]] == [
        "user", "assistant", "user", "assistant",
    ]


@pytest.mark.asyncio
async def test_history_is_isolated_per_user(auth_client, db_session):
    _, token_a = await make_user(db_session)
    _, token_b = await make_user(db_session)

    first = await auth_client.post(
        "/ask/search",
        json={"query": "private", "stream": False},
        headers=bearer(token_a,
    ),
    )
    conversation_id = first.json()["answer"]["conversation_id"]

    foreign = await auth_client.get(
        f"/history/{conversation_id}", headers=bearer(token_b)
    )
    assert foreign.status_code == 404

    foreign_ask = await auth_client.post(
        "/ask/search",
        json={"query": "q", "stream": False, "conversation_id": conversation_id},
        headers=bearer(token_b,
    ),
    )
    assert foreign_ask.status_code == 404


@pytest.mark.asyncio
async def test_delete_conversation(auth_client, db_session):
    _, token = await make_user(db_session)
    first = await auth_client.post(
        "/ask/search",
        json={"query": "temp", "stream": False},
        headers=bearer(token,
    ),
    )
    conversation_id = first.json()["answer"]["conversation_id"]

    deleted = await auth_client.delete(
        f"/history/{conversation_id}", headers=bearer(token)
    )
    assert deleted.status_code == 204
    assert (
        await auth_client.get(f"/history/{conversation_id}", headers=bearer(token))
    ).status_code == 404


# <----- /user routes: self-or-admin visibility (PLAN §1 audit fix) ----->


@pytest.mark.asyncio
async def test_user_routes_reject_anonymous(auth_client, db_session):
    user, _ = await make_user(db_session)

    assert (await auth_client.get(f"/user/{user.id}")).status_code == 401
    assert (await auth_client.get(f"/user/nn/{user.nickname}")).status_code == 401
    patch = await auth_client.patch(f"/user/{user.id}", json={"nickname": "nope"})
    assert patch.status_code == 401
    assert (await auth_client.delete(f"/user/{user.id}")).status_code == 401


@pytest.mark.asyncio
async def test_foreign_users_are_hidden_not_forbidden(auth_client, db_session):
    _, owner_token = await make_user(db_session)
    other, _ = await make_user(db_session)

    read = await auth_client.get(f"/user/{other.id}", headers=bearer(owner_token))
    assert read.status_code == 404
    by_nick = await auth_client.get(
        f"/user/nn/{other.nickname}", headers=bearer(owner_token)
    )
    assert by_nick.status_code == 404
    patch = await auth_client.patch(
        f"/user/{other.id}",
        json={"nickname": "stolen"},
        headers=bearer(owner_token),
    )
    assert patch.status_code == 404
    deleted = await auth_client.delete(
        f"/user/{other.id}", headers=bearer(owner_token)
    )
    assert deleted.status_code == 404


@pytest.mark.asyncio
async def test_self_can_read_and_update_own_profile(auth_client, db_session):
    user, token = await make_user(db_session)
    headers = bearer(token)

    me = await auth_client.get(f"/user/{user.id}", headers=headers)
    assert me.status_code == 200
    assert me.json()["id"] == str(user.id)

    by_nick = await auth_client.get(f"/user/nn/{user.nickname}", headers=headers)
    assert by_nick.status_code == 200

    renamed = await auth_client.patch(
        f"/user/{user.id}",
        json={"nickname": user.nickname},
        headers=headers,
    )
    assert renamed.status_code == 200


@pytest.mark.asyncio
async def test_admin_can_manage_other_users(auth_client, db_session):
    _, admin_token = await make_user(db_session, role_name="admin")
    other, _ = await make_user(db_session)

    read = await auth_client.get(f"/user/{other.id}", headers=bearer(admin_token))
    assert read.status_code == 200

    deleted = await auth_client.delete(
        f"/user/{other.id}", headers=bearer(admin_token)
    )
    assert deleted.status_code == 204


# <----- Vector store in the pipeline: the turn must survive (report F-01) ----->


class StoringEmbedder:
    """Offline embedder: deterministic vectors, really engages the chunk store.

    When an embedder resolves, the search pipeline stores chunks mid-request
    (a commit on the shared session) before the answer turn is persisted —
    the exact sequence that used to 500 the JSON request and silently drop
    the turn.
    """

    dims = 4

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


@pytest.mark.asyncio
async def test_search_with_vector_store_persists_the_turn(auth_client, db_session):
    user, token = await make_user(db_session)

    # Vector store in the hot path, exactly like a deploy with an embedder.
    app.dependency_overrides[get_optional_embedder] = lambda: StoringEmbedder()

    response = await auth_client.post(
        "/ask/search",
        json={"query": "storable question", "stream": False},
        headers=bearer(token),
    )
    assert response.status_code == 200, response.text
    conversation_id = response.json()["answer"].get("conversation_id")
    assert conversation_id

    # The user/assistant turn was persisted despite the mid-pipeline commit.
    listed = await auth_client.get("/history", headers=bearer(token))
    assert listed.status_code == 200
    listed_ids = [c["id"] for c in listed.json()]
    assert conversation_id in listed_ids

    fetched = await auth_client.get(
        f"/history/{conversation_id}", headers=bearer(token)
    )
    assert fetched.status_code == 200
    assert [m["role"] for m in fetched.json()["messages"]] == ["user", "assistant"]
