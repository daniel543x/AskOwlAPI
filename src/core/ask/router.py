from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel

from ...modules.identity.auth.security import get_current_user
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
    searching: ISearchProvider = Depends(get_search_provider),
    model: BaseChatModel = Depends(get_llm_client),
    scraper: IScraper = Depends(get_scraper),
    ranker: IRanker = Depends(get_ranker),
):
    return StreamingResponse(
        sse_search_generator(query, searching, model, scraper, ranker),
        media_type="text/event-stream",
    )


@router.get("/research")
async def ask_research(
    user: str = Depends(get_current_user),
    model: BaseChatModel = Depends(get_llm_client),
):
    message = user
    message = model.invoke("Hey!")
    return {"message": message}


@router.get("/deep_research")
async def ask_deep_research():
    return
