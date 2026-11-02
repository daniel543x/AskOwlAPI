"""Integration tests: the memory repository and model-deletion cleanup."""

import pytest

from src.modules.llm_provider.model import ModelCatalogEntry, ModelProvider
from src.modules.llm_provider.repository import ModelRepository
from src.modules.profile.model import UserModelSettings
from src.modules.profile.repository import ModelSettingsRepository

pytestmark = pytest.mark.integration

USER_ID = None  # set per test


async def _commit(session):
    await session.commit()


async def test_settings_repo_upsert_get_overwrites(db_session):
    import uuid

    from src.modules.identity.user.model import UserCreate
    from src.modules.identity.user.service import create_user

    row = await create_user(
        db_session,
        UserCreate(
            email=f"{uuid.uuid4().hex[:8]}@e.com",
            nickname=f"u{uuid.uuid4().hex[:6]}",
            password="Sup3rSecret!pass",
        ),
    )
    user_id = row.id
    provider = ModelProvider(name=f"p_{user_id.hex[:6]}", provider_type="ollama")
    db_session.add(provider)
    await db_session.flush()
    entry = ModelCatalogEntry(
        provider_id=provider.id, kind="chat", model_name="m", is_active=True
    )
    db_session.add(entry)
    await db_session.commit()
    await db_session.refresh(entry)

    repo = ModelSettingsRepository(db_session)
    await repo.upsert(user_id, entry.id, True, "low")
    row = await repo.get(user_id, entry.id)
    assert row is not None
    assert (row.thinking, row.effort) == (True, "low")
    await repo.upsert(user_id, entry.id, None, None)
    row = await repo.get(user_id, entry.id)
    assert row is not None
    assert (row.thinking, row.effort) == (None, None)


async def test_deleting_a_model_removes_memory_rows(db_session):
    import uuid

    from src.modules.identity.user.model import UserCreate
    from src.modules.identity.user.service import create_user

    row = await create_user(
        db_session,
        UserCreate(
            email=f"{uuid.uuid4().hex[:8]}@e.com",
            nickname=f"u{uuid.uuid4().hex[:6]}",
            password="Sup3rSecret!pass",
        ),
    )
    user_id = row.id
    provider = ModelProvider(name=f"p_{user_id.hex[:6]}", provider_type="ollama")
    db_session.add(provider)
    await db_session.flush()
    entry = ModelCatalogEntry(
        provider_id=provider.id, kind="chat", model_name="gone", is_active=True
    )
    db_session.add(entry)
    await db_session.commit()
    await db_session.refresh(entry)
    db_session.add(
        UserModelSettings(user_id=user_id, model_id=entry.id, thinking=True)
    )
    await db_session.commit()

    await ModelRepository(db_session).delete_model(entry.id)
    leftover = await db_session.get(
        UserModelSettings, (user_id, entry.id)
    )
    assert leftover is None


async def test_profile_restore_through_real_repos(db_session):
    """End-to-end: switch away and back through the real repositories."""
    import uuid

    from httpx import ASGITransport, AsyncClient
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from src.app import app
    from src.modules.embedding.factory import get_optional_embedder
    from src.modules.identity.auth.service import create_access_token
    from src.modules.identity.user.model import UserCreate
    from src.modules.identity.user.service import create_user
    from src.modules.llm_provider.dependency import (
        get_llm_client,
        get_turn_resources,
    )
    from src.modules.llm_provider.model import ModelCatalogEntry as Entry
    from src.modules.llm_provider.model import ModelKind
    from src.modules.llm_provider.model import ModelProvider as Provider
    from src.modules.owl_reranker.factory import get_ranker
    from src.modules.scrapy.factory import get_scraper
    from src.modules.search_engine.factory import get_search_provider

    row = await create_user(
        db_session,
        UserCreate(
            email=f"{uuid.uuid4().hex[:8]}@e.com",
            nickname=f"u{uuid.uuid4().hex[:6]}",
            password="Sup3rSecret!pass",
        ),
    )
    token = create_access_token({"sub": str(row.id)})
    headers = {"Authorization": f"Bearer {token}"}

    provider = Provider(name=f"p_{row.id.hex[:6]}", provider_type="ollama")
    db_session.add(provider)
    await db_session.flush()
    entry = Entry(
        provider_id=provider.id,
        kind=ModelKind.CHAT.value,
        model_name="capable",
        is_active=True,
        required_role="user",
        capabilities=["tools", "thinking"],
    )
    db_session.add(entry)
    await db_session.commit()
    await db_session.refresh(entry)

    # The seed provider config knows its own source types; the model
    # resolutions happen through the GET/PUT routes below.
    async_overrides = app.dependency_overrides
    async_overrides[get_llm_client] = lambda: None
    async_overrides[get_turn_resources] = lambda: None
    async_overrides[get_search_provider] = lambda: None
    async_overrides[get_scraper] = lambda: None
    async_overrides[get_ranker] = lambda: None
    async_overrides[get_optional_embedder] = lambda: None

    # Route the app's session dependency at db_session directly.
    async def override_get_session():
        yield db_session

    from src.tools.db import get_session

    async_overrides[get_session] = override_get_session

    engine_session_factory = async_sessionmaker(
        bind=db_session.bind, class_=async_sessionmaker  # not used
    )
    del engine_session_factory

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as ac:
            r1 = await ac.put(
                "/profile/preferences",
                json={"default_model_id": str(entry.id), "thinking": True},
                headers=headers,
            )
            assert r1.status_code == 200, r1.text
            r2 = await ac.put(
                "/profile/preferences", json={}, headers=headers
            )
            assert r2.status_code == 200, r2.text
            r3 = await ac.put(
                "/profile/preferences",
                json={"default_model_id": str(entry.id)},
                headers=headers,
            )
            assert r3.status_code == 200, r3.text
            assert r3.json()["resolved"]["thinking"] is True
    finally:
        async_overrides.clear()
