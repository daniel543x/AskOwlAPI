from sqlmodel.ext.asyncio.session import AsyncSession

from .modules.identity.role.service import init_roles
from .modules.llm_provider.repository import get_repository
from .modules.scrapy.init_scrapy import init_scrapy_settings
from .tools.db import engine, init_db


# <----- Init all: setings, db table, objects, etc. ----->
async def init():
    await init_db(engine)

    async with AsyncSession(engine) as session:
        await init_roles(session=session)
        await init_scrapy_settings(session)
        repo = get_repository(session)
        await repo.init_default_llm_config()
