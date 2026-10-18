import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from .factory import ScraperFactory
from .model import ScrapySettings

logger = logging.getLogger(__name__)


async def init_scrapy_settings(session: AsyncSession):
    registered_scrapers: dict[str, type] = ScraperFactory.REGISTRY
    DEFAULT_SCRAPER_NAME = "trafilatura"

    statement = select(ScrapySettings)
    result = await session.execute(statement)
    db_scrapers = result.scalars().all()
    db_scrapers_map = {scraper.name: scraper for scraper in db_scrapers}

    current_defaults: list[ScrapySettings] = [s for s in db_scrapers if s.is_default]

    needs_commit = False

    if len(current_defaults) == 0:
        logger.info(
            "No default scraper found. Setting 'trafilatura' as default and enabled."
        )
        trafilatura_entry = db_scrapers_map.get(DEFAULT_SCRAPER_NAME)
        if trafilatura_entry:
            trafilatura_entry.is_default = True
            trafilatura_entry.enabled = True
            needs_commit = True
    elif len(current_defaults) > 1:
        logger.warning(
            "Inconsistent state: %d default scrapers found. "
            "Fixing to have only '%s' as default.",
            len(current_defaults),
            DEFAULT_SCRAPER_NAME,
        )
        for scraper in current_defaults:
            if scraper.name == DEFAULT_SCRAPER_NAME:
                scraper.is_default = True
            else:
                scraper.is_default = False
        needs_commit = True

    for scraper_name in registered_scrapers:
        if scraper_name not in db_scrapers_map:
            logger.info("Creating new scraper entry for '%s'.", scraper_name)
            is_default = (
                scraper_name == DEFAULT_SCRAPER_NAME and len(current_defaults) == 0
            )

            new_scraper = ScrapySettings(
                name=scraper_name,
                is_default=is_default,
                enabled=is_default,
                api_key=None,
                settings={},
            )
            session.add(new_scraper)
            needs_commit = True

    if needs_commit:
        await session.commit()
        logger.info("Database changes committed.")
    else:
        logger.info("No changes needed.")

    logger.info("Database initialization for scrapers finished successfully.")
