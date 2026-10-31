"""Integration tests: soft delete hides rows without destruction."""

import uuid

import pytest
import pytest_asyncio

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def auth_client(db_session):
    from httpx import ASGITransport, AsyncClient

    from src.app import app
    from src.modules.identity.auth.service import create_access_token
    from src.modules.identity.user.model import UserCreate
    from src.modules.identity.role.service import assign_role_to_user
    from src.modules.identity.user.service import create_user
    from src.tools.db import get_session

    row = await create_user(
        db_session,
        UserCreate(
            email=f"{uuid.uuid4().hex[:8]}@e.com",
            nickname=f"u{uuid.uuid4().hex[:6]}",
            password="Sup3rSecret!pass",
        ),
    )
    assert await assign_role_to_user(db_session, row.id, "research")
    token = create_access_token({"sub": str(row.id)})
    headers = {"Authorization": f"Bearer {token}"}

    async def override_get_session():
        yield db_session

    app.dependency_overrides[get_session] = override_get_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as ac:
        yield ac, row, headers
    app.dependency_overrides.clear()


async def _conversation(db_session, user_id, persona_id=None):
    from src.modules.history.model import Conversation

    row = Conversation(user_id=user_id, title="t", persona_id=persona_id)
    db_session.add(row)
    await db_session.commit()
    await db_session.refresh(row)
    return row


async def test_deleted_conversation_hidden_from_reads(auth_client, db_session):
    ac, user, headers = auth_client
    row = await _conversation(db_session, user.id)

    deleted = await ac.delete(f"/history/{row.id}", headers=headers)
    assert deleted.status_code == 204

    listed = (await ac.get("/history", headers=headers)).json()
    assert listed == []
    assert (await ac.get(f"/history/{row.id}", headers=headers)).status_code == 404

    from sqlmodel import select

    from src.modules.history.model import Conversation

    kept = (await db_session.exec(select(Conversation))).all()
    assert len(kept) == 1 and kept[0].deleted_at is not None  # row survives


async def test_deleted_persona_hidden_from_reads(auth_client, db_session):
    ac, user, headers = auth_client
    created = await ac.post(
        "/personas", json={"name": "doomed"}, headers=headers
    )
    assert created.status_code == 201, created.text
    persona_id = created.json()["id"]

    deleted = await ac.delete(f"/personas/{persona_id}", headers=headers)
    assert deleted.status_code == 204

    assert (await ac.get("/personas", headers=headers)).json() == []
    assert (
        await ac.get(f"/personas/{persona_id}", headers=headers)
    ).status_code == 404

    from sqlmodel import select

    from src.modules.personas.model import Persona

    kept = (await db_session.exec(select(Persona))).all()
    assert len(kept) == 1 and kept[0].deleted_at is not None


async def test_soft_persona_delete_keeps_conversations(auth_client, db_session):
    ac, user, headers = auth_client
    from src.modules.history.model import Conversation
    from src.modules.personas.model import Persona

    persona = Persona(user_id=user.id, name="keeper")
    db_session.add(persona)
    await db_session.commit()
    await db_session.refresh(persona)
    row = await _conversation(db_session, user.id, persona.id)

    deleted = await ac.delete(f"/personas/{persona.id}", headers=headers)
    assert deleted.status_code == 204

    listed = (await ac.get("/history", headers=headers)).json()
    mapped = {c["id"]: c["persona_id"] for c in listed}
    assert mapped[str(row.id)] == str(persona.id)
