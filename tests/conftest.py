import os
from collections.abc import AsyncGenerator
from urllib.parse import quote_plus

# Provide deterministic, offline-safe settings BEFORE any src import:
# the app refuses to start without a valid MASTER_KEY.
os.environ.setdefault(
    "MASTER_KEY", "s2w6bWtjpHynwD8xBHyDzICIRvJ1ioxmhP39k6iBKDI="
)
# Hermetic tests: the settings singleton (created below) must not inherit
# the developer shell or the local .env. Debug mode, for instance, would
# make the pipeline emit extra SSE events that break the documented
# happy-path sequence a protocol test asserts.
os.environ["LOG_LEVEL"] = "INFO"
os.environ["DEBUG"] = "false"

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from src.app import app  # noqa: E402

# Import all models so create_all sees every table.
from src.modules.api_keys.model import ApiKey  # noqa: E402, F401
from src.modules.chunk_store.model import Chunk  # noqa: E402, F401
from src.modules.history.model import Conversation, Message  # noqa: E402, F401
from src.modules.identity.auth.model import RefreshToken  # noqa: E402, F401
from src.modules.identity.role.model import Role, UserRole  # noqa: E402, F401
from src.modules.identity.role.service import init_roles  # noqa: E402
from src.modules.identity.user.model import User  # noqa: E402, F401
from src.modules.llm_provider.model import (  # noqa: E402, F401
    DefaultSettings,
    ModelCatalogEntry,
    ModelProvider,
    UserModelPreference,
)
from src.modules.personas.model import Persona  # noqa: E402, F401
from src.modules.profile.model import UserPreferences  # noqa: E402, F401
from src.modules.scrapy.model import ScrapedPage, ScrapySettings  # noqa: E402, F401
from src.modules.search_engine.model import SearchProviderConfig  # noqa: E402, F401
from src.tools.db import get_session, init_db  # noqa: E402


def _integration_database_url() -> str:
    """URL of the dedicated integration-test database.

    NEVER the application database: the fixture drops all tables on
    teardown. The test DB is a separate database (created on demand) on the
    same server the POSTGRES_* variables point at, and a hard guard refuses
    to run if it would resolve to the configured application database.
    """
    test_db = os.getenv("POSTGRES_DB_TEST", "askowl_test")
    app_db = os.getenv("POSTGRES_DB", "database")
    if test_db == app_db:
        raise RuntimeError(
            "Integration tests would use the application database "
            f"'{app_db}' (POSTGRES_DB_TEST must differ from POSTGRES_DB; "
            "the fixture drops all tables)."
        )
    return (
        f"postgresql+asyncpg://{os.getenv('POSTGRES_USER', 'postgres')}:"
        f"{quote_plus(os.getenv('POSTGRES_PASSWORD', 'postgres'))}@"
        f"{os.getenv('POSTGRES_SERVER', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/{test_db}"
    )


async def _ensure_test_database() -> str:
    """Create the integration-test database if it does not exist yet."""
    from sqlalchemy import text

    url = _integration_database_url()
    server_url, test_db = url.rsplit("/", 1)
    # Connect through the maintenance database to create the test DB if
    # needed; Postgres always requires a database on connect.
    engine = create_async_engine(f"{server_url}/postgres", isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as conn:
            exists = await conn.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": test_db},
            )
            if not exists:
                await conn.execute(text(f'CREATE DATABASE "{test_db}"'))
        return url
    finally:
        await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession]:
    """Transactional database session. Requires a live Postgres (integration)."""
    url = await _ensure_test_database()
    test_engine = create_async_engine(url, echo=False)
    session_factory = async_sessionmaker(
        bind=test_engine, class_=AsyncSession, expire_on_commit=False
    )

    await init_db(engine=test_engine)

    # Seed default roles the way the app lifespan does, so routes that
    # require them (e.g. user creation) behave as in production.
    async with session_factory() as session:
        await init_roles(session=session)

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
