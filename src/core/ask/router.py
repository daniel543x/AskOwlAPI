"""Ask routes: the `search` mode.

POST (not GET): a search turn persists a conversation (history side
effects), so the request is a body-carrying POST. Model selection params
(`model_id`, `embedder_id`, `reranker_id`, `search_provider`) stay on the
query string via the shared dependencies. SSE streaming works identically
over POST.
"""

import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ...modules.attachments.service import resolve_for_ask
from ...modules.decision.base import IDecisionMaker
from ...modules.decision.factory import get_optional_decision_maker
from ...modules.embedding.base import IEmbedder
from ...modules.embedding.factory import get_optional_embedder
from ...modules.identity.auth.security import require_scope
from ...modules.identity.user.model import User
from ...modules.jobs.factory import get_job_queue
from ...modules.jobs.gates import (
    GateBusy,
    TurnResources,
    acquire_gates,
    provider_busy_detail,
    release_gates,
)
from ...modules.jobs.model import Job
from ...modules.llm_provider.dependency import get_llm_client, get_turn_resources
from ...modules.owl_reranker.base import IRanker
from ...modules.owl_reranker.factory import get_ranker
from ...modules.scrapy.factory import get_scraper
from ...modules.scrapy.providers.base import IScraper
from ...modules.search_engine.factory import (
    get_declared_source_types,
    get_search_provider,
)
from ...modules.search_engine.providers.base import ISearchProvider
from ...modules.search_engine.source_types import ensure_source_types
from ...tools.db import engine, get_session
from ...tools.settings import settings
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
    source_types: list[str] = Field(
        default=[],
        description=(
            "Source types for this turn (PR 11). Empty = the provider's "
            "own defaults; validated against GET /search/sources."
        ),
    )
    attachment_ids: list[uuid.UUID] = Field(
        default=[],
        description=(
            "Uploaded files for this turn (PR 15). Must belong to the "
            "continued conversation; see POST /conversations/{id}/"
            "attachments."
        ),
    )
    background: bool = Field(
        default=False,
        description=(
            "Run this turn as a background job (PR 12): responds 202 with "
            "a job id; the answer is not streamed."
        ),
    )
    thinking: bool | None = Field(
        default=None,
        description=(
            "Reasoning-mode control (decision 11). null = model default; "
            "requires the model's 'thinking' capability (422 otherwise)."
        ),
    )
    effort: Literal["low", "medium", "high", "max"] | None = Field(
        default=None,
        description=(
            "Reasoning-effort control (decision 11). null = model default; "
            "requires the 'effort' capability and a supported level "
            "(422 with a coded error otherwise)."
        ),
    )


@router.post("/search", dependencies=[Depends(require_scope("search"))])
async def ask_search(
    request: SearchRequest,
    searching: ISearchProvider = Depends(get_search_provider),
    declared_source_types: list[str] = Depends(get_declared_source_types),
    scraper: IScraper = Depends(get_scraper),
    ranker: IRanker = Depends(get_ranker),
    embedder: IEmbedder | None = Depends(get_optional_embedder),
    decision_maker: IDecisionMaker | None = Depends(get_optional_decision_maker),
    user: User = Depends(_search_scope),
    session: AsyncSession = Depends(get_session),
    model: BaseChatModel = Depends(get_llm_client),
    turn: TurnResources = Depends(get_turn_resources),
):
    ensure_source_types(request.source_types, declared_source_types)
    attachments = await resolve_for_ask(
        session, user.id, request.conversation_id, request.attachment_ids
    )
    if request.background:
        return await _enqueue_search_turn(
            request, model, searching, scraper, ranker, embedder,
            decision_maker, user, turn,
        )
    if settings.BUSY_POLICY == "reject":
        try:
            await acquire_gates(turn.keys, timeout=0)
        except GateBusy as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=provider_busy_detail(),
            ) from e
        release_gates(turn.keys)
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
                scope=request.scope,
                recency_bias=request.recency_bias,
                source_types=request.source_types,
                turn=turn,
                attachments=[
                    {"id": a.id, "filename": a.filename,
                     "conversation_id": a.conversation_id}
                    for a in attachments
                ],
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
        scope=request.scope,
        recency_bias=request.recency_bias,
        source_types=request.source_types,
        turn=turn,
        attachments=[
            {"id": a.id, "filename": a.filename,
             "conversation_id": a.conversation_id}
            for a in attachments
        ],
    )
    return {"answer": answer}


async def _enqueue_search_turn(
    request: SearchRequest,
    model: BaseChatModel,
    searching: ISearchProvider,
    scraper: IScraper,
    ranker: IRanker,
    embedder: IEmbedder | None,
    decision_maker: IDecisionMaker | None,
    user: User,
    turn: TurnResources,
) -> JSONResponse:
    """Auto-convert a busy foreground ask into a queued job (PR 12)."""
    from sqlalchemy.ext.asyncio import AsyncSession

    job = Job(
        user_id=user.id,
        kind="search",
        payload={
            "query": request.query,
            "conversation_id": str(request.conversation_id)
            if request.conversation_id
            else None,
        },
    )

    async def runner() -> str:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            result = await search_json_adapter(
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
                scope=request.scope,
                recency_bias=request.recency_bias,
                source_types=request.source_types,
                turn=turn,
            )
        return (result.get("answer") or "")[:4000]

    saved = await get_job_queue().submit(job, runner)
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "job_id": str(saved.id),
            "kind": saved.kind,
            "status": saved.status,
            "queue_position": saved.queue_position,
        },
    )
