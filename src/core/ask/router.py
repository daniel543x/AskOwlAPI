import json

from fastapi import APIRouter, Depends

from ...modules.llm_provider.factory import get_llm_client
from ...modules.llm_provider.providers.base import LLMProviderBase
from ...modules.owl_reranker.owl_reranker import OwlRanker
from ...modules.scrapy.factory import get_scraper
from ...modules.scrapy.providers.base import IScraper
from ...modules.search_engine.factory import get_search_provider
from ...modules.search_engine.providers.base import ISearchProvider
from .services.chunking import chunk_markdown_data
from .services.search import make_answer_for_search, make_query_for_search_engine

router = APIRouter(prefix="/ask", tags=["Asks"])


@router.get("/search")
async def ask_search(
    query: str,
    searching: ISearchProvider = Depends(get_search_provider),
    model: LLMProviderBase = Depends(get_llm_client),
    scraper: IScraper = Depends(get_scraper),
):
    # --- LLM building query ---
    search_query = await model.generate(make_query_for_search_engine(query))
    search_query = json.loads(search_query)

    # --- searching ---
    sources = await searching.search(query=search_query.get("search_query"))

    # --- ranking ---
    ranker = OwlRanker()
    sources = ranker.rank_web_search(query, sources)

    # --- scraping ---
    content = await scraper.scrape(sources)

    # --- chunking ---
    if content:
        content = chunk_markdown_data(content)

    # --- ranking ---
    content = ranker.rank_chunks(query, content, 20)

    # --- LLM answer building ---
    answer = await model.generate(make_answer_for_search(query, content))

    return {
        "answer": answer,
        "search_query": search_query,
        "sources": sources,
    }


@router.get("/research")
async def ask_research():
    return


@router.get("/deep_research")
async def ask_deep_research():
    return
