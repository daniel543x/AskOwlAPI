from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel

from ...modules.llm_provider.dependency import get_llm_client
from ...modules.owl_reranker.base import IRanker
from ...modules.owl_reranker.factory import get_ranker
from ...modules.scrapy.factory import get_scraper
from ...modules.scrapy.providers.base import IScraper
from ...modules.search_engine.factory import get_search_provider
from ...modules.search_engine.providers.base import ISearchProvider
from .services.search import sse_search_generator

router = APIRouter(prefix="/ask", tags=["Asks"])


@router.get("/search")
async def ask_search(
    query: str = Query(..., description="User query."),
    stream: bool = Query(
        True, description="Enable/Disenable streaming response. Default True."
    ),
    searching: ISearchProvider = Depends(get_search_provider),
    model: BaseChatModel = Depends(get_llm_client),
    scraper: IScraper = Depends(get_scraper),
    ranker: IRanker = Depends(get_ranker),
):
    if stream:
        return StreamingResponse(
            sse_search_generator(query, searching, model, scraper, ranker),
            media_type="text/event-stream",
        )
    else:
        return {"answer": "answer"}


@router.get("/research")
async def ask_research(
    query: str = Query(..., description="User query."),
    searching: ISearchProvider = Depends(get_search_provider),
    model: BaseChatModel = Depends(get_llm_client),
    scraper: IScraper = Depends(get_scraper),
    ranker: IRanker = Depends(get_ranker),
):
    sources = await searching.search(query)
    message = ranker.rank_web_search(query, sources, 5)

    return {"message": message}


@router.get("/deep_research")
async def ask_deep_research(
    query: str = Query(..., description="User query."),
    searching: ISearchProvider = Depends(get_search_provider),
    model: BaseChatModel = Depends(get_llm_client),
    scraper: IScraper = Depends(get_scraper),
    ranker: IRanker = Depends(get_ranker),
):
    return
