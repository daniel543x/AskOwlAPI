"""Research and agent routes (elevated modes, protocol v2 extensions).

- `GET /ask/research` — planner -> per-sub-question search -> cited report.
- `GET /ask/agent` — autonomous agent loop over the shared tools, optionally
  constrained by one of the caller's personas (`persona_id`).

Both are gated on an elevated role (pro/admin) per PLAN section 0.5 and
share the ask pipeline's dependency wiring.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.language_models import BaseChatModel
from sqlmodel.ext.asyncio.session import AsyncSession

from ...modules.chunk_store.repository import ChunkRepository
from ...modules.decision.base import IDecisionMaker
from ...modules.decision.factory import get_optional_decision_maker
from ...modules.embedding.base import IEmbedder
from ...modules.embedding.factory import get_optional_embedder
from ...modules.identity.auth.security import (
    get_current_elevated_user,
    require_scope,
)
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
from ...modules.personas.repository import PersonaRepository
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
from ..tools.toolbox import AskToolbox
from .services.pipeline import (
    agent_sse_adapter,
    make_budget,
    research_json_adapter,
    research_sse_adapter,
)

router = APIRouter(prefix="/ask", tags=["Research"])


@router.get(
    "/research",
    dependencies=[
        Depends(get_current_elevated_user),
        Depends(require_scope("research")),
    ],
)
async def ask_research(
    query: str = Query(..., description="User query."),
    stream: bool = Query(
        True, description="Enable/Disable streaming response. Default True."
    ),
    searching: ISearchProvider = Depends(get_search_provider),
    declared_source_types: list[str] = Depends(get_declared_source_types),
    model: BaseChatModel = Depends(get_llm_client),
    scraper: IScraper = Depends(get_scraper),
    ranker: IRanker = Depends(get_ranker),
    embedder: IEmbedder | None = Depends(get_optional_embedder),
    session: AsyncSession = Depends(get_session),
    source_types: list[str] = Query(default=[]),
    turn: TurnResources = Depends(get_turn_resources),
    background: bool = Query(default=False),
    user: User = Depends(get_current_elevated_user),
):
    ensure_source_types(source_types, declared_source_types)
    if background:
        return await _enqueue_research_turn(
            query, searching, model, scraper, ranker, embedder,
            session, user, source_types, turn,
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
    if stream:
        return StreamingResponse(
            research_sse_adapter(
                query,
                searching,
                model,
                scraper,
                ranker,
                embedder=embedder,
                session=session,
                source_types=source_types,
                turn=turn,
            ),
            media_type="text/event-stream",
        )

    return await research_json_adapter(
        query,
        searching,
        model,
        scraper,
        ranker,
        embedder=embedder,
        session=session,
        source_types=source_types,
        turn=turn,
    )


@router.get(
    "/agent",
    dependencies=[
        Depends(get_current_elevated_user),
        Depends(require_scope("agent")),
    ],
)
async def ask_agent(
    goal: str = Query(..., description="Goal the agent should accomplish."),
    persona_id: uuid.UUID | None = Query(
        default=None, description="Persona constraining the agent (own only)."
    ),
    searching: ISearchProvider = Depends(get_search_provider),
    model: BaseChatModel = Depends(get_llm_client),
    scraper: IScraper = Depends(get_scraper),
    ranker: IRanker = Depends(get_ranker),
    embedder: IEmbedder | None = Depends(get_optional_embedder),
    decision_maker: IDecisionMaker | None = Depends(get_optional_decision_maker),
    user: User = Depends(get_current_elevated_user),
    session: AsyncSession = Depends(get_session),
    declared_source_types: list[str] = Depends(get_declared_source_types),
    source_types: list[str] = Query(default=[]),
    turn: TurnResources = Depends(get_turn_resources),
):
    ensure_source_types(source_types, declared_source_types)
    persona = None
    if persona_id is not None:
        persona = await PersonaRepository(session).get_for_user(user.id, persona_id)
        if persona is None:
            raise HTTPException(status_code=404, detail="Persona not found.")

    chunk_repo = ChunkRepository(session) if session is not None else None
    toolbox = AskToolbox(
        searching,
        scraper,
        ranker,
        embedder,
        chunk_repo,
        allowed_domains=persona.allowed_domains if persona else None,
        allowed_engines=persona.allowed_engines if persona else None,
        decision_maker=decision_maker,
        source_types=source_types,
    )

    return StreamingResponse(
        agent_sse_adapter(
            goal,
            model,
            toolbox,
            make_budget(persona.default_parameters if persona else None),
            persona=persona,
            turn=turn,
        ),
        media_type="text/event-stream",
    )


async def _enqueue_research_turn(
    query: str,
    searching: ISearchProvider,
    model: BaseChatModel,
    scraper: IScraper,
    ranker: IRanker,
    embedder: IEmbedder | None,
    session: AsyncSession,
    user: User,
    source_types: list[str],
    turn: TurnResources,
) -> JSONResponse:
    """Auto-convert a busy foreground research run into a job (PR 12)."""
    from sqlalchemy.ext.asyncio import AsyncSession

    job = Job(user_id=user.id, kind="research", payload={"query": query})

    async def runner() -> str:
        async with AsyncSession(engine, expire_on_commit=False) as job_session:
            result = await research_json_adapter(
                query,
                searching,
                model,
                scraper,
                ranker,
                embedder=embedder,
                session=job_session,
                source_types=source_types,
                turn=turn,
            )
        return (result.get("report") or "")[:4000]

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
