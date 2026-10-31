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
        # conversations (webapp #8). The FK is matched structurally (table,
        # column, referenced table) rather than by name: existing databases
        # carry the constraint under non-default names such as
        # fk_conversation_persona_id, which a name-based guard never sees.
        await conn.execute(
            text(
                "DO $$\n"
                "DECLARE legacy_fk text;\n"
                "BEGIN\n"
                "    LOOP\n"
                "        SELECT c.conname INTO legacy_fk\n"
                "        FROM pg_constraint c\n"
                "        JOIN pg_attribute a ON a.attrelid = c.conrelid\n"
                "          AND a.attnum = ANY (c.conkey)\n"
                "        WHERE c.conrelid = 'conversation'::regclass\n"
                "          AND c.contype = 'f'\n"
                "          AND c.confrelid = 'persona'::regclass\n"
                "          AND a.attname = 'persona_id'\n"
                "          AND c.confdeltype <> 'n'\n"
                "        LIMIT 1;\n"
                "        EXIT WHEN legacy_fk IS NULL;\n"
                "        EXECUTE format(\n"
                "            'ALTER TABLE conversation DROP CONSTRAINT %I',\n"
                "            legacy_fk);\n"
                "    END LOOP;\n"
                "    IF NOT EXISTS (\n"
                "        SELECT 1 FROM pg_constraint c\n"
                "        JOIN pg_attribute a ON a.attrelid = c.conrelid\n"
                "          AND a.attnum = ANY (c.conkey)\n"
                "        WHERE c.conrelid = 'conversation'::regclass\n"
                "          AND c.contype = 'f'\n"
                "          AND c.confrelid = 'persona'::regclass\n"
                "          AND a.attname = 'persona_id'\n"
                "          AND c.confdeltype = 'n'\n"
                "    ) THEN\n"
                "        ALTER TABLE conversation "
                "ADD CONSTRAINT conversation_persona_id_fkey "
                "FOREIGN KEY (persona_id) REFERENCES persona(id) "
                "ON DELETE SET NULL;\n"
                "    END IF;\n"
                "END $$;"
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
