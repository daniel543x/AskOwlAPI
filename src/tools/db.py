from collections.abc import AsyncGenerator
from urllib.parse import quote_plus

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from .settings import settings


def build_database_url() -> str:
    """Async Postgres URL from settings (never hardcoded)."""
    return (
        f"postgresql+asyncpg://"
        f"{settings.POSTGRES_USER}:"
        f"{quote_plus(settings.POSTGRES_PASSWORD)}@"
        f"{settings.POSTGRES_SERVER}:"
        f"{settings.POSTGRES_PORT}/"
        f"{settings.POSTGRES_DB}"
    )


# Setup DataBase
engine = create_async_engine(build_database_url(), pool_pre_ping=True, echo=False)


# Function for init db tables
async def init_db(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        # The chunk store needs pgvector; the compose db image ships it.
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(SQLModel.metadata.create_all)


# For Dependency
async def get_session() -> AsyncGenerator[AsyncSession]:
    # expire_on_commit=False matches the test harness (conftest) and keeps
    # post-commit attribute reads (token rotation, API-key last_used,
    # admin block) from triggering sync lazy refreshes, which raise
    # MissingGreenlet in async context — the trap PLAN Phase 3 documents.
    async with AsyncSession(engine, expire_on_commit=False) as session:
        yield session
