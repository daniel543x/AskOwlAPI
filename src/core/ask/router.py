"""Ask routes: the `search` mode.

POST (not GET): a search turn persists a conversation (history side
effects), so the request is a body-carrying POST. Model selection params
(`model_id`, `embedder_id`, `reranker_id`, `search_provider`) stay on the
query string via the shared dependencies. SSE streaming works identically
over POST.
"""

import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ...modules.decision.base import IDecisionMaker
from ...modules.decision.factory import get_optional_decision_maker
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


class SearchRequest(BaseModel):
    """Body of POST /ask/search."""

    query: str = Field(description="User query.")
    stream: bool = Field(
        default=True, description="Stream the answer as SSE. Default true."
    )
    conversation_id: uuid.UUID | None = Field(
        default=None, description="Continue an existing conversation (own only).",
    )


@router.post("/search", dependencies=[Depends(require_scope("search"))])
async def ask_search(
    request: SearchRequest,
    searching: ISearchProvider = Depends(get_search_provider),
    model: BaseChatModel = Depends(get_llm_client),
    scraper: IScraper = Depends(get_scraper),
    ranker: IRanker = Depends(get_ranker),
    embedder: IEmbedder | None = Depends(get_optional_embedder),
    decision_maker: IDecisionMaker | None = Depends(get_optional_decision_maker),
    user: User = Depends(_search_scope),
    session: AsyncSession = Depends(get_session),
):
    if request.stream:
        return StreamingResponse(
            search_sse_adapter(
                request.query,
                searching,
                model,
                scraper,
                ranker,
                embedder=embedder,
                session=session,
                user_id=user.id,
                conversation_id=request.conversation_id,
                decision_maker=decision_maker,
            ),
            media_type="text/event-stream",
        )

    answer = await search_json_adapter(
        request.query,
        searching,
        model,
        scraper,
        ranker,
        embedder=embedder,
        session=session,
        user_id=user.id,
        conversation_id=request.conversation_id,
        decision_maker=decision_maker,
    )
    return {"answer": answer}
