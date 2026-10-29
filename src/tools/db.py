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
        # create_all cannot alter existing tables and Alembic has not landed
        # yet (PLAN Phase 6), so FK changes ship as this guarded bootstrap
        # migration. conversation.persona_id must be ON DELETE SET NULL
        # ("n" in pg_constraint.confdeltype) so deleting a persona keeps its
        # conversations (webapp #8).
        await conn.execute(
            text(
                "DO $$\n"
                "BEGIN IF EXISTS (\n"
                "    SELECT 1 FROM pg_constraint\n"
                "    WHERE conname = 'conversation_persona_id_fkey'\n"
                "      AND conrelid = 'conversation'::regclass\n"
                "      AND confdeltype NOT IN ('n')\n"
                ") THEN\n"
                "    ALTER TABLE conversation "
                "DROP CONSTRAINT conversation_persona_id_fkey;\n"
                "    ALTER TABLE conversation "
                "ADD CONSTRAINT conversation_persona_id_fkey "
                "FOREIGN KEY (persona_id) REFERENCES persona(id) "
                "ON DELETE SET NULL;\n"
                "END IF; END $$;"
            )
        )


# For Dependency
async def get_session() -> AsyncGenerator[AsyncSession]:
    # expire_on_commit=False matches the test harness (conftest) and keeps
    # post-commit attribute reads (token rotation, API-key last_used,
    # admin block) from triggering sync lazy refreshes, which raise
    # MissingGreenlet in async context — the trap PLAN Phase 3 documents.
    async with AsyncSession(engine, expire_on_commit=False) as session:
        yield session
