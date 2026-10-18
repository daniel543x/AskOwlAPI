import asyncio

from sqlmodel.ext.asyncio.session import AsyncSession

from .modules.identity.role.service import init_roles
from .modules.llm_provider.repository import ModelRepository
from .modules.scrapy.init_scrapy import init_scrapy_settings
from .tools.db import build_database_url, engine
from .tools.migrations import run_migrations


# <----- Init all: settings, db tables, objects, etc. ----->
async def init():
    # Schema bootstrap: Alembic migrations (create_all is tests-only now).
    await asyncio.to_thread(run_migrations, build_database_url())

    async with AsyncSession(engine) as session:
        await init_roles(session=session)
        await init_scrapy_settings(session)
        repo = ModelRepository(session)
        await repo.bootstrap()
