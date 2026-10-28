
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from ...tools.db import get_session
from ...tools.settings import settings
from .model import ScrapySettings
from .page_cache import PageCacheRepository, PageCachingScraper
from .providers.base import IScraper
from .providers.trafilatura import TrafilturaScraper


# If not specified,
# prepare the default from the database;
# if specified, prepare the specified one.
class ScraperFactory:
    REGISTRY: dict[str, type[IScraper]] = {
        "trafilatura": TrafilturaScraper,
    }

    @classmethod
    async def create(
        cls, session: AsyncSession, scraper_name: str | None = None
    ) -> IScraper:
        if scraper_name:
            query = select(ScrapySettings).where(
                ScrapySettings.name == scraper_name, ScrapySettings.enabled
            )
        else:
            query = select(ScrapySettings).where(
                ScrapySettings.is_default, ScrapySettings.enabled
            )

        result = await session.execute(query)
        config = result.scalar_one_or_none()

        if not config:
            if scraper_name:
                raise ValueError(f"Scraper '{scraper_name}' not found or not enabled")
            else:
                raise ValueError("No default scraper configured")

        scraper_class = cls.REGISTRY.get(config.name)
        if not scraper_class:
            raise ValueError(f"Scraper '{config.name}' not implemented in code")

        instance = scraper_class(api_key=config.api_key, config=config.settings)

        if settings.PAGE_CACHE:
            # Per-URL content cache (WU-10): shares the request session, so
            # its failures degrade to a full fetch without breaking the turn.
            instance = PageCachingScraper(instance, PageCacheRepository(session))

        return instance


async def get_scraper(
    session: AsyncSession = Depends(get_session),
):
    return await ScraperFactory.create(session)
