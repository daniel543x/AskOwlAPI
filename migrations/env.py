"""Alembic environment (async engine, SQLModel metadata, pgvector).

URLs are never hardcoded: the database URL is built from the same
settings as the application. Offline/programmatic runs may pass an
explicit URL via -x url=... (used by tests and the app bootstrap).
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from src.modules.api_keys.model import ApiKey  # noqa: F401
from src.modules.chunk_store.model import Chunk  # noqa: F401
from src.modules.history.model import Conversation, Message  # noqa: F401
from src.modules.identity.role.model import Role, UserRole  # noqa: F401
from src.modules.identity.user.model import User  # noqa: F401
from src.modules.llm_provider.model import (  # noqa: F401
    DefaultSettings,
    ModelCatalogEntry,
    ModelProvider,
    UserModelPreference,
)
from src.modules.personas.model import Persona  # noqa: F401
from src.modules.scrapy.model import ScrapedPage, ScrapySettings  # noqa: F401
from src.modules.search_engine.model import SearchProviderConfig  # noqa: F401
from src.tools.db import SQLModel, build_database_url

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import every model (above) so metadata knows all tables.
target_metadata = SQLModel.metadata


def _database_url() -> str:
    # Priority: -x url=... > sqlalchemy.url main option > app settings.
    x_arg = context.get_x_argument(as_dictionary=True).get("url")
    if x_arg:
        return x_arg
    main = config.get_main_option("sqlalchemy.url")
    if main and not main.startswith("driver://"):
        return main
    return build_database_url()


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _database_url()
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
