"""Programmatic Alembic migrations (the app's schema bootstrap).

Startup runs `alembic upgrade head` against the configured database; the
migrations under `migrations/versions/` are the schema's source of truth.
Databases created by the old `create_all` bootstrap (pre-Alembic) are
adopted by stamping the current revision instead of failing. `init_db`
(create_all) remains only as the test harness bootstrap.
"""

import asyncio
import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
ALEMBIC_INI = _ROOT / "alembic.ini"


def _config(database_url: str) -> Config:
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


async def _is_pre_alembic_database(database_url: str) -> bool:
    """True when app tables exist but Alembic never managed the schema."""
    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as conn:
            managed = await conn.scalar(
                text("SELECT to_regclass('public.alembic_version')")
            )
            if managed is not None:
                return False
            has_app_tables = await conn.scalar(
                text("SELECT to_regclass('public.user')")
            )
            return has_app_tables is not None
    finally:
        await engine.dispose()


def run_migrations(database_url: str) -> None:
    """Upgrade the database to head (call via asyncio.to_thread)."""
    config = _config(database_url)
    if asyncio.run(_is_pre_alembic_database(database_url)):
        logger.info(
            "Adopting a pre-Alembic database: stamping revision head "
            "(the existing create_all schema is kept as-is)."
        )
        command.stamp(config, "head")
        return
    command.upgrade(config, "head")
