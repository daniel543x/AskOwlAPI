import uuid

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from sqlalchemy.ext.asyncio import AsyncSession

from ...modules.embedding.base import IEmbedder
from ...modules.embedding.factory import get_optional_embedder
from ...modules.identity.auth.security import require_scope
from ...modules.identity.user.model import User
from ...modules.llm_provider.dependency import get_llm_client
from ...modules.owl_reranker.base import IRanker
from ...modules.owl_reranker.factory import get_ranker
from ...modules.scrapy.factory import get_scraper
from ...modules.scrapy.providers.base import IScraper
from ...modules.search_engine.factory import get_search_provider
from ...modules.search_engine.providers.base import ISearchProvider
from ...tools.db import get_session
from .services.search import search_json_adapter, search_sse_adapter

router = APIRouter(prefix="/ask", tags=["Asks"])

_search_scope = require_scope("search")


@router.get("/search", dependencies=[Depends(require_scope("search"))])
async def ask_search(
    query: str = Query(..., description="User query."),
    stream: bool = Query(
        True, description="Enable/Disenable streaming response. Default True."
    ),
    conversation_id: uuid.UUID | None = Query(
        default=None, description="Continue an existing conversation (own only)."
    ),
    searching: ISearchProvider = Depends(get_search_provider),
    model: BaseChatModel = Depends(get_llm_client),
    scraper: IScraper = Depends(get_scraper),
    ranker: IRanker = Depends(get_ranker),
    embedder: IEmbedder | None = Depends(get_optional_embedder),
    user: User = Depends(_search_scope),
    session: AsyncSession = Depends(get_session),
):
    if stream:
        return StreamingResponse(
            search_sse_adapter(
                query,
                searching,
                model,
                scraper,
                ranker,
                embedder=embedder,
                session=session,
                user_id=user.id,
                conversation_id=conversation_id,
            ),
            media_type="text/event-stream",
        )
    else:
        answer = await search_json_adapter(
            query,
            searching,
            model,
            scraper,
            ranker,
            embedder=embedder,
            session=session,
            user_id=user.id,
            conversation_id=conversation_id,
        )
        return {"answer": answer}
