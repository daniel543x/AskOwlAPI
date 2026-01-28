from typing import Optional

from fastapi import Depends

from ....tools.db import get_session
from ...llm_provider.providers.ollama import OllamaProvider
from ...scrapy.factory import get_scraper
from ...scrapy.providers.base import IScraper

# from ...scrapy.providers.trafilatura import TrafilturaScraper
from ...search_engine.providers.searxng import SearXNGProvider


async def search(
    query: str,
    search_provider: Optional[str],
    llm_provider: Optional[str],
    session=Depends(get_session),
    scraper: IScraper = Depends(get_scraper),
):
    # --- Factory ---
    searching = SearXNGProvider()
    # scraper = get_scraper()  # TrafilturaScraper()
    model = OllamaProvider("llama3.1:8b")

    # --- searching ---
    sources = await searching.search(query=query, max_results=10)

    # --- scraping ---
    content = []

    for x in sources:
        markdown = await scraper.scrape(x)
        if markdown is not None:
            content.append(markdown)

    # --- LLM answer building ---
    answer = await model.chat(
        f'You are great researcher. Make answer to this user query "{query}". Use this content from web search: {content}. Return answer in '
    )
    return answer
