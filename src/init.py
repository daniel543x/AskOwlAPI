import asyncio
import logging
from typing import Any, cast

from sqlalchemy import text
from sqlmodel.ext.asyncio.session import AsyncSession

from .modules.identity.role.service import init_roles
from .modules.llm_provider.repository import ModelRepository
from .modules.scrapy.init_scrapy import init_scrapy_settings
from .tools.db import build_database_url, engine
from .tools.migrations import run_migrations
from .tools.settings import settings

logger = logging.getLogger(__name__)


async def evict_old_chunks() -> None:
    """Purge chunks older than CHUNK_TTL_DAYS at boot (AR-8 task 3).

    The chunk store is a cache, not an archive: without eviction it grows
    unbounded (F-06). 0 disables. Failures log and never block startup.
    """
    if settings.CHUNK_TTL_DAYS <= 0:
        return
    try:
        async with AsyncSession(engine) as session:
            result = await session.execute(
                text(
                    "DELETE FROM chunk WHERE created_at < "
                    "now() - (:days * interval '1 day')"
                ),
                {"days": settings.CHUNK_TTL_DAYS},
            )
            await session.commit()
            deleted = cast("Any", result).rowcount or 0
            if deleted:
                logger.info(
                    "Chunk eviction removed %d chunks older than %d days.",
                    deleted,
                    settings.CHUNK_TTL_DAYS,
                )
    except Exception:
        logger.warning("Chunk eviction failed; continuing startup.", exc_info=True)


# <----- Init all: settings, db tables, objects, etc. ----->
async def init():
    # Schema bootstrap: Alembic migrations (create_all is tests-only now).
    await asyncio.to_thread(run_migrations, build_database_url())
    await evict_old_chunks()

    async with AsyncSession(engine) as session:
        await init_roles(session=session)
        await init_scrapy_settings(session)
        repo = ModelRepository(session)
        await repo.bootstrap()
