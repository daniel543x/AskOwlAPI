from typing import Dict, List

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from .factory import ScraperFactory
from .model import ScrapySettings


async def init_scrapy_settings(session: AsyncSession):
    registered_scrapers: Dict[str, type] = ScraperFactory.REGISTRY
    DEFAULT_SCRAPER_NAME = "trafilatura"

    statement = select(ScrapySettings)
    result = await session.execute(statement)
    db_scrapers = result.scalars().all()
    db_scrapers_map = {scraper.name: scraper for scraper in db_scrapers}

    current_defaults: List[ScrapySettings] = [s for s in db_scrapers if s.is_default]

    needs_commit = False

    if len(current_defaults) == 0:
        print("No default scraper found. Setting 'trafilatura' as default and enabled.")
        trafilatura_entry = db_scrapers_map.get(DEFAULT_SCRAPER_NAME)
        if trafilatura_entry:
            trafilatura_entry.is_default = True
            trafilatura_entry.enabled = True
            needs_commit = True
    elif len(current_defaults) > 1:
        print(
            f"Inconsistent state: {len(current_defaults)} default scrapers found. "
            f"Fixing to have only '{DEFAULT_SCRAPER_NAME}' as default."
        )
        for scraper in current_defaults:
            if scraper.name == DEFAULT_SCRAPER_NAME:
                scraper.is_default = True
            else:
                scraper.is_default = False
        needs_commit = True

    for scraper_name in registered_scrapers:
        if scraper_name not in db_scrapers_map:
            print(f"Creating new scraper entry for '{scraper_name}'.")
            is_default = (
                scraper_name == DEFAULT_SCRAPER_NAME and len(current_defaults) == 0
            )

            new_scraper = ScrapySettings(
                name=scraper_name,
                is_default=is_default,
                enabled=is_default,
                api_key=None,
                config={},
            )
            session.add(new_scraper)
            needs_commit = True

    if needs_commit:
        await session.commit()
        print("Database changes committed.")
    else:
        print("No changes needed.")

    print("Database initialization for scrapers finished successfully.")
