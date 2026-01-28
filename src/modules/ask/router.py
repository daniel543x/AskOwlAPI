from fastapi import APIRouter, Depends

from ..llm_provider.providers.base import LLMProviderBase
from ..llm_provider.providers.factory import get_llm_client
from ..scrapy.factory import get_scraper
from ..scrapy.providers.base import IScraper
from ..search_engine.factory import get_search_provider
from ..search_engine.providers.base import ISearchProvider

router = APIRouter(prefix="/ask", tags=["Asks"])


@router.get("/search")
async def ask_search(
    query: str,
    searching: ISearchProvider = Depends(get_search_provider),
    model: LLMProviderBase = Depends(get_llm_client),
    scraper: IScraper = Depends(get_scraper),
):
    # --- searching ---
    sources = await searching.search(query=query, max_results=10)

    # --- scraping ---
    content = []
    for x in sources:
        markdown = await scraper.scrape(x)
        if markdown is not None:
            content.append(markdown)

    # --- LLM answer building ---
    answer = await model.generate(
        f'You are great researcher. Make answer to this user query "{query}". Use this content from web search: {content}. Return answer in '
    )

    return {"answer": answer}


@router.get("/research")
async def ask_research():
    return


@router.get("/auto")
async def ask_autonomic():
    return
