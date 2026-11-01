"""History reads: persona_id exposure and the persona filter (integration)."""

import uuid
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.app import app
from src.modules.history.model import Conversation, Message
from src.modules.identity.auth.service import create_access_token
from src.modules.identity.user.model import UserCreate
from src.modules.identity.user.service import create_user
from src.modules.personas.model import Persona
from src.tools.db import get_session

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def auth_client(db_session):
    async def override_get_session():
        yield db_session

    app.dependency_overrides[get_session] = override_get_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


async def make_user(db_session: AsyncSession) -> tuple[str, Any]:
    suffix = uuid.uuid4().hex[:8]
    user = await create_user(
        db_session,
        UserCreate(
            email=f"{suffix}@example.com",
            nickname=f"user_{suffix}",
            password="Sup3rSecret!pass",
        ),
    )
    return create_access_token({"sub": str(user.id)}), user


async def add_persona(db_session: AsyncSession, user_id: uuid.UUID) -> Persona:
    persona = Persona(user_id=user_id, name="p")
    db_session.add(persona)
    await db_session.commit()
    await db_session.refresh(persona)
    return persona


async def add_conversation(
    db_session: AsyncSession, user_id: uuid.UUID, persona_id: uuid.UUID | None = None
) -> Conversation:
    conversation = Conversation(user_id=user_id, title="t", persona_id=persona_id)
    db_session.add(conversation)
    await db_session.commit()
    await db_session.refresh(conversation)
    return conversation


@pytest.mark.asyncio
async def test_conversation_read_exposes_persona_id(auth_client, db_session):
    token, user = await make_user(db_session)
    headers = {"Authorization": f"Bearer {token}"}
    persona = await add_persona(db_session, user.id)
    scoped = await add_conversation(db_session, user.id, persona.id)
    plain = await add_conversation(db_session, user.id)

    listed = await auth_client.get("/history", headers=headers)
    assert listed.status_code == 200, listed.text
    by_id = {c["id"]: c["persona_id"] for c in listed.json()}
    assert by_id[str(scoped.id)] == str(persona.id)
    assert by_id[str(plain.id)] is None

    detail = await auth_client.get(f"/history/{scoped.id}", headers=headers)
    assert detail.status_code == 200, detail.text
    assert detail.json()["persona_id"] == str(persona.id)


@pytest.mark.asyncio
async def test_history_filters_by_persona_id(auth_client, db_session):
    token, user = await make_user(db_session)
    headers = {"Authorization": f"Bearer {token}"}
    persona = await add_persona(db_session, user.id)
    scoped = await add_conversation(db_session, user.id, persona.id)
    await add_conversation(db_session, user.id)

    filtered = await auth_client.get(
        f"/history?persona_id={persona.id}", headers=headers
    )
    assert filtered.status_code == 200, filtered.text
    assert [c["id"] for c in filtered.json()] == [str(scoped.id)]


@pytest.mark.asyncio
async def test_history_filter_is_user_scoped(auth_client, db_session):
    token_a, user_a = await make_user(db_session)
    token_b, user_b = await make_user(db_session)
    persona_b = await add_persona(db_session, user_b.id)
    foreign_scoped = await add_conversation(db_session, user_a.id, persona_b.id)
    await add_conversation(db_session, user_a.id)

    as_a = await auth_client.get(
        f"/history?persona_id={persona_b.id}",
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert as_a.status_code == 200, as_a.text
    assert [c["id"] for c in as_a.json()] == [str(foreign_scoped.id)]

    as_b = await auth_client.get(
        f"/history?persona_id={persona_b.id}",
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert as_b.status_code == 200, as_b.text
    assert as_b.json() == []


@pytest.mark.asyncio
async def test_history_messages_carry_turn_meta(auth_client, db_session):
    """The 'mode · model' caption must survive reloads (API-NOTES #11)."""
    token, user = await make_user(db_session)
    headers = {"Authorization": f"Bearer {token}"}
    conversation = await add_conversation(db_session, user.id)
    db_session.add(
        Message(
            conversation_id=conversation.id,
            role="user",
            content="what is rust?",
            meta={"mode": "search", "model_label": "qwen3.5"},
        )
    )
    db_session.add(
        Message(conversation_id=conversation.id, role="assistant", content="safe.")
    )
    await db_session.commit()

    detail = await auth_client.get(
        f"/history/{conversation.id}", headers=headers
    )
    assert detail.status_code == 200, detail.text
    messages = detail.json()["messages"]
    assert messages[0]["meta"] == {
        "mode": "search",
        "model_label": "qwen3.5",
    }
    assert messages[1]["meta"] is None
