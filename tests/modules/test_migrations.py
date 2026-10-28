"""Alembic migration validation (integration: live Postgres).

Guards the upgrade and downgrade paths on a clean scratch database, per
PLAN Phase 6, and asserts the migrated schema matches the model metadata.
"""

import asyncio
import uuid

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

from src.tools.db import build_database_url

pytestmark = pytest.mark.integration

ALEMBIC_INI = "alembic.ini"

EXPECTED_TABLES = {
    "alembic_version",
    "apikey",
    "attachment",
    "chunk",
    "conversation",
    "defaultsettings",
    "message",
    "modelcatalogentry",
    "modelprovider",
    "persona",
    "refreshtoken",
    "role",
    "scraped_page",
    "scrapysettings",
    "searchproviderconfig",
    "user",
    "usermodelpreference",
    "job",
    "userpreferences",
    "userrole",
}


def _config(url: str) -> Config:
    config = Config(ALEMBIC_INI)
    config.set_main_option("script_location", "migrations")
    return config


def _database_url(database: str) -> str:
    base = build_database_url()
    return base.rsplit("/", 1)[0] + f"/{database}"


@pytest.fixture
def scratch_db():
    database = f"askowl_migrate_{uuid.uuid4().hex[:8]}"
    yield database
    # Teardown happens inside the test (drop via maintenance connection).
    return database


async def _create_database(name: str) -> None:
    base = build_database_url()
    server_url = base.rsplit("/", 1)[0]
    engine = create_async_engine(
        f"{server_url}/postgres", isolation_level="AUTOCOMMIT"
    )
    async with engine.connect() as conn:
        await conn.execute(text(f'CREATE DATABASE "{name}"'))
    await engine.dispose()


async def _drop_database(name: str) -> None:
    base = build_database_url()
    server_url = base.rsplit("/", 1)[0]
    engine = create_async_engine(
        f"{server_url}/postgres", isolation_level="AUTOCOMMIT"
    )
    async with engine.connect() as conn:
        await conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
    await engine.dispose()


async def _tables(database: str) -> set[str]:
    engine = create_async_engine(_database_url(database))
    async with engine.connect() as conn:
        rows = await conn.scalars(
            text("SELECT tablename FROM pg_tables WHERE schemaname='public'")
        )
        tables = set(rows)
    await engine.dispose()
    return tables


@pytest.mark.asyncio
async def test_upgrade_downgrade_upgrade_on_clean_database(scratch_db):
    await _create_database(scratch_db)
    url = _database_url(scratch_db)
    try:
        config = _config(url)

        config.set_main_option("sqlalchemy.url", url)
        await asyncio.to_thread(command.upgrade, config, "head")
        assert await _tables(scratch_db) == EXPECTED_TABLES

        config.set_main_option("sqlalchemy.url", url)
        await asyncio.to_thread(command.downgrade, config, "base")
        assert await _tables(scratch_db) == {"alembic_version"}

        config.set_main_option("sqlalchemy.url", url)
        await asyncio.to_thread(command.upgrade, config, "head")
        assert await _tables(scratch_db) == EXPECTED_TABLES
    finally:
        await _drop_database(scratch_db)


@pytest.mark.asyncio
async def test_migration_adds_a_row_roundtrip(scratch_db):
    """The migrated schema actually serves the app's repositories."""
    await _create_database(scratch_db)
    url = _database_url(scratch_db)
    try:
        config = _config(url)
        config.set_main_option("sqlalchemy.url", url)
        await asyncio.to_thread(command.upgrade, config, "head")

        engine = create_async_engine(url)
        async with engine.begin() as conn:
            await conn.execute(
                text("INSERT INTO \"user\" (id, email, nickname, "
                     "password_hash, is_active, created_at) VALUES ("
                     "gen_random_uuid(), 'migrate@example.com', 'migrate', "
                     "'x', true, now())")
            )
        async with engine.connect() as conn:
            count = await conn.scalar(text("SELECT count(*) FROM \"user\""))
        assert count == 1
        await engine.dispose()
    finally:
        await _drop_database(scratch_db)


def test_model_metadata_covers_migrated_tables():
    """Every migrated table comes from the SQLModel metadata (no drift)."""
    model_tables = {table.lower() for table in SQLModel.metadata.tables}
    assert EXPECTED_TABLES - {"alembic_version"} == model_tables
