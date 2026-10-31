"""Integration tests: soft delete hides rows without destruction."""

import uuid

import pytest
import pytest_asyncio

pytestmark = pytest.mark.integration

from tests.core_ask.fakes import (  # noqa: E402
    FakeRanker,
    FakeSearchProvider,
    ScriptedToolModel,
    make_scraper,
    tool_call,
)

ask_provider = FakeSearchProvider()
ask_model = ScriptedToolModel(
    responses=[tool_call("CitedAnswer", {"answer": "x"})]
)
ask_scraper = make_scraper()
ask_ranker = FakeRanker()




@pytest_asyncio.fixture
async def auth_client(db_session):
    from httpx import ASGITransport, AsyncClient

    from src.app import app
    from src.modules.identity.auth.service import create_access_token
    from src.modules.identity.role.service import assign_role_to_user
    from src.modules.identity.user.model import UserCreate
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


@pytest_asyncio.fixture
async def ask_client(db_session, auth_client):
    """auth_client plus the ask-route fakes wired like the persona harness."""
    from src.app import app
    from src.modules.embedding.factory import get_optional_embedder
    from src.modules.llm_provider.dependency import get_llm_client
    from src.modules.owl_reranker.factory import get_ranker
    from src.modules.scrapy.factory import get_scraper
    from src.modules.search_engine.factory import get_search_provider
    from tests.core_ask.fakes import (
        FakeRanker,
        FakeSearchProvider,
        ScriptedToolModel,
        make_scraper,
        tool_call,
    )

    app.dependency_overrides[get_search_provider] = lambda: FakeSearchProvider()
    app.dependency_overrides[get_scraper] = lambda: make_scraper()
    app.dependency_overrides[get_ranker] = lambda: FakeRanker()
    app.dependency_overrides[get_optional_embedder] = lambda: None
    app.dependency_overrides[get_llm_client] = lambda: ScriptedToolModel(
        responses=[tool_call("CitedAnswer", {"answer": "kept."})]
    )
    yield auth_client
    for name in (
        get_search_provider,
        get_scraper,
        get_ranker,
        get_optional_embedder,
        get_llm_client,
    ):
        app.dependency_overrides.pop(name, None)


async def test_deleted_conversation_rejects_continuation(
    ask_client, db_session
):
    ac, user, headers = ask_client
    row = await _conversation(db_session, user.id)

    deleted = await ac.delete(f"/history/{row.id}", headers=headers)
    assert deleted.status_code == 204

    response = await ac.post(
        "/ask/search",
        json={"query": "continue?", "conversation_id": str(row.id), "stream": False},
        headers=headers,
    )
    assert response.status_code == 404, response.text

    from sqlmodel import select

    from src.modules.history.model import Message

    assert (await db_session.exec(select(Message))).all() == []


async def test_deleted_persona_rejects_ask(ask_client, db_session):
    ac, user, headers = ask_client
    from src.modules.personas.model import Persona

    persona = Persona(user_id=user.id, name="gone")
    db_session.add(persona)
    await db_session.commit()
    await db_session.refresh(persona)

    deleted = await ac.delete(f"/personas/{persona.id}", headers=headers)
    assert deleted.status_code == 204

    response = await ac.post(
        "/ask/persona",
        json={"query": "q?", "persona_id": str(persona.id), "depth": "quick"},
        headers=headers,
    )
    assert response.status_code == 404, response.text


async def test_background_runner_fails_cleanly(ask_client, db_session):
    """A queued turn targeting a deleted conversation fails, sanitized."""
    import asyncio

    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )

    from src.modules.jobs.model import Job, JobStatus
    from src.modules.jobs.queue import JobQueue
    from tests.conftest import _integration_database_url

    ac, user, headers = ask_client
    row = await _conversation(db_session, user.id)
    await ac.delete(f"/history/{row.id}", headers=headers)

    test_engine = create_async_engine(_integration_database_url())
    session_factory = async_sessionmaker(
        bind=test_engine, class_=AsyncSession, expire_on_commit=False
    )


    async def runner() -> str:
        from src.core.ask.services.search import search_json_adapter

        async with session_factory() as session:
            await search_json_adapter(
                "deleted target?",
                ask_provider,
                ask_model,
                ask_scraper,
                ask_ranker,
                session=session,
                user_id=row.id,
                conversation_id=row.id,
            )
        return "unreachable"

    queue = JobQueue(session_factory=session_factory)
    queue.start()
    try:
        job = Job(user_id=row.id, kind="search", payload={"query": "x"})
        await queue.submit(job, runner)
        for _ in range(200):
            if job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
                break
            await asyncio.sleep(0.05)
        assert job.status == JobStatus.FAILED
        assert job.error == "Job failed with HTTPException."
    finally:
        await queue.stop()
        await test_engine.dispose()
