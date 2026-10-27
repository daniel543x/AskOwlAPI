"""Ask routes: the `search` mode.

POST (not GET): a search turn persists a conversation (history side
effects), so the request is a body-carrying POST. Model selection params
(`model_id`, `embedder_id`, `reranker_id`, `search_provider`) stay on the
query string via the shared dependencies. SSE streaming works identically
over POST.
"""

import uuid
from typing import Literal

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
    max_urls: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Cap on URLs selected for this turn (WU-6). Omit for the "
            "TOP_N_URLS default; clamped server-side to TOP_N_URLS_MAX."
        ),
    )
    query_variations: int = Field(
        default=1,
        ge=1,
        description=(
            "Rule-based query fan-out for this turn (WU-2). Variants run "
            "against the search provider and merge into one ranked pool; "
            "clamped server-side to QUERY_VARIATIONS_MAX."
        ),
    )
    cache: Literal["on", "bypass"] = Field(
        default="on",
        description=(
            '"bypass" forces fresh search + scrape for this turn (WU-10), '
            "e.g. for evals or admin debugging."
        ),
    )
    scope: Literal["global", "conversation", "request"] = Field(
        default="global",
        description=(
            "Retrieval scope (WU-5). \"global\" (default, today's behavior) "
            'retrieves from the whole chunk store; "conversation" restricts '
            'vector candidates to this conversation (+ legacy unscoped '
            'rows); "request" restricts retrieval to this turn\'s fetched '
            "pages."
        ),
    )
    recency_bias: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Recency boost for chunk ranking (WU-11): 0 = neutral, 1 = "
            "strongest. Clamped server-side to RECENCY_BIAS_MAX."
        ),
    )
    hybrid_alpha: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Hybrid retrieval blend (WU-4): 1.0 = pure vector (default, "
            "today's behavior), 0.0 = pure lexical full-text. In between, "
            "both legs fuse by alpha-weighted reciprocal rank fusion "
            "before the cross-encoder rerank."
        ),
    )
    rewrite: Literal["standalone", "simple", "none"] = Field(
        default="standalone",
        description=(
            'Query rewriting (WU-1). "standalone" resolves pronouns and '
            'omissions against the conversation history; "simple" is '
            'history-free rewriting; "none" skips the rewriting LLM call '
            "entirely (eval/tool callers)."
        ),
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
                max_urls=request.max_urls,
                query_variations=request.query_variations,
                cache=request.cache,
                rewrite=request.rewrite,
                scope=request.scope,
                recency_bias=request.recency_bias,
                hybrid_alpha=request.hybrid_alpha,
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
        max_urls=request.max_urls,
        query_variations=request.query_variations,
        cache=request.cache,
        rewrite=request.rewrite,
        scope=request.scope,
        recency_bias=request.recency_bias,
        hybrid_alpha=request.hybrid_alpha,
    )
    return {"answer": answer}
