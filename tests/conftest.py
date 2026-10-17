import os
from collections.abc import AsyncGenerator
from urllib.parse import quote_plus

# Provide deterministic, offline-safe settings BEFORE any src import:
# the app refuses to start without a valid MASTER_KEY.
os.environ.setdefault(
    "MASTER_KEY", "s2w6bWtjpHynwD8xBHyDzICIRvJ1ioxmhP39k6iBKDI="
)

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from src.app import app  # noqa: E402
from src.modules.chunk_store.model import Chunk  # noqa: E402, F401
from src.modules.identity.role.model import Role, UserRole  # noqa: E402, F401

# Import all models so create_all sees every table.
from src.modules.identity.user.model import User  # noqa: E402, F401
from src.modules.llm_provider.model import (  # noqa: E402, F401
    DefaultSettings,
    ModelCatalogEntry,
    ModelProvider,
    UserModelPreference,
)
from src.modules.scrapy.model import ScrapySettings  # noqa: E402, F401
from src.modules.search_engine.model import SearchProviderConfig  # noqa: E402, F401
from src.tools.db import get_session, init_db  # noqa: E402


def _integration_database_url() -> str:
    return (
        f"postgresql+asyncpg://{os.getenv('POSTGRES_USER', 'postgres')}:"
        f"{quote_plus(os.getenv('POSTGRES_PASSWORD', 'postgres'))}@"
        f"{os.getenv('POSTGRES_SERVER', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB', 'postgres')}"
    )


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession]:
    """Transactional database session. Requires a live Postgres (integration)."""
    test_engine = create_async_engine(_integration_database_url(), echo=False)
    session_factory = async_sessionmaker(
        bind=test_engine, class_=AsyncSession, expire_on_commit=False
    )

    await init_db(engine=test_engine)

    async with session_factory() as session:
        yield session

    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    await test_engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient]:
    """HTTP client wired to the app with the DB session overridden (integration)."""

    async def override_get_session():
        yield db_session

    app.dependency_overrides[get_session] = override_get_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()
