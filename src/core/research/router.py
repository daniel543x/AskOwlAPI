"""Research and agent routes (elevated modes, protocol v2 extensions).

- `GET /ask/research` — planner -> per-sub-question search -> cited report.
- `GET /ask/agent` — autonomous agent loop over the shared tools, optionally
  constrained by one of the caller's personas (`persona_id`).

Both are gated on an elevated role (pro/admin) per PLAN section 0.5 and
share the ask pipeline's dependency wiring.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from sqlalchemy.ext.asyncio import AsyncSession

from ...modules.chunk_store.repository import ChunkRepository
from ...modules.embedding.base import IEmbedder
from ...modules.embedding.factory import get_optional_embedder
from ...modules.identity.auth.security import get_current_elevated_user
from ...modules.identity.user.model import User
from ...modules.llm_provider.dependency import get_llm_client
from ...modules.owl_reranker.base import IRanker
from ...modules.owl_reranker.factory import get_ranker
from ...modules.personas.repository import PersonaRepository
from ...modules.scrapy.factory import get_scraper
from ...modules.scrapy.providers.base import IScraper
from ...modules.search_engine.factory import get_search_provider
from ...modules.search_engine.providers.base import ISearchProvider
from ...tools.db import get_session
from ..tools.toolbox import AskToolbox
from .services.pipeline import (
    agent_sse_adapter,
    make_budget,
    research_json_adapter,
    research_sse_adapter,
)

router = APIRouter(prefix="/ask", tags=["Research"])


@router.get("/research", dependencies=[Depends(get_current_elevated_user)])
async def ask_research(
    query: str = Query(..., description="User query."),
    stream: bool = Query(
        True, description="Enable/Disable streaming response. Default True."
    ),
    searching: ISearchProvider = Depends(get_search_provider),
    model: BaseChatModel = Depends(get_llm_client),
    scraper: IScraper = Depends(get_scraper),
    ranker: IRanker = Depends(get_ranker),
    embedder: IEmbedder | None = Depends(get_optional_embedder),
    session: AsyncSession = Depends(get_session),
):
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
    )


@router.get("/agent")
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
    user: User = Depends(get_current_elevated_user),
    session: AsyncSession = Depends(get_session),
):
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
    )

    return StreamingResponse(
        agent_sse_adapter(
            goal,
            model,
            toolbox,
            make_budget(persona.default_parameters if persona else None),
            persona=persona,
        ),
        media_type="text/event-stream",
    )
