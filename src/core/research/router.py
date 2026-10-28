"""Research and agent routes (elevated modes, protocol v2 extensions).

- `GET /ask/research` — planner -> per-sub-question search -> cited report.
- `GET /ask/agent` — autonomous agent loop over the shared tools, optionally
  constrained by one of the caller's personas (`persona_id`).

Both are gated on an elevated role (pro/admin) per PLAN section 0.5 and
share the ask pipeline's dependency wiring.
"""

import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel
from sqlmodel.ext.asyncio.session import AsyncSession

from src.core.ask.services.search import _resolve_conversation, _save_turn
from src.core.research.services.agent import agent_logic

from ...modules.attachments.service import (
    ensure_vision,
    fetch_image_blobs,
    resolve_for_ask,
)
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
from ...modules.notes.repository import NoteRepository
from ...modules.notes.store import NoteStore
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


def get_note_store(
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_elevated_user),
) -> NoteStore:
    """Run-scoped note writer for research-grade turns (PR 18)."""
    return NoteStore(
        NoteRepository(session), user.id
    )


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
    attachment_ids: list[uuid.UUID] = Query(default=[]),
    note_store: NoteStore = Depends(get_note_store),
):
    return await _research_turn(
        query, stream, searching, declared_source_types, model, scraper,
        ranker, embedder, session, source_types, turn, background, user,
        attachment_ids, note_store, scout=False, reflect=False,
    )


@router.get(
    "/deep_research",
    dependencies=[
        Depends(get_current_elevated_user),
        Depends(require_scope("research")),
    ],
)
async def ask_deep_research(
    query: str = Query(..., description="Research goal (v1 name: agent)."),
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
    attachment_ids: list[uuid.UUID] = Query(default=[]),
    note_store: NoteStore = Depends(get_note_store),
):
    """The deep-research preset (PR 19): scout + reflect + notes.

    High budget via the shared Budget; renamed from `agent` in v3
    (PR 21). The old `/ask/agent` loop stays as a deprecated alias.
    """
    return await _research_turn(
        query, stream, searching, declared_source_types, model, scraper,
        ranker, embedder, session, source_types, turn, background, user,
        attachment_ids, note_store, scout=True, reflect=True,
    )


class PersonaAskRequest(BaseModel):
    """Body of POST /ask/persona (PR 20 / decision 10)."""

    query: str
    persona_id: uuid.UUID  # required: missing = 422 by schema
    depth: Literal["quick", "research", "deep"] | None = None
    stream: bool = True
    background: bool = False
    attachment_ids: list[uuid.UUID] = []
    source_types: list[str] = []
    conversation_id: uuid.UUID | None = None


def _resolve_depth(request: PersonaAskRequest, persona) -> str:
    """Request > persona.default_parameters > settings default (clamped)."""
    if request.depth:
        return request.depth
    configured = (persona.default_parameters or {}).get("depth")
    if configured in ("quick", "research", "deep"):
        return configured
    return settings.PERSONA_DEFAULT_DEPTH


@router.post(
    "/persona",
    dependencies=[
        Depends(get_current_elevated_user),
        Depends(require_scope("agent")),
    ],
)
async def ask_persona(
    request: PersonaAskRequest,
    searching: ISearchProvider = Depends(get_search_provider),
    declared_source_types: list[str] = Depends(get_declared_source_types),
    model: BaseChatModel = Depends(get_llm_client),
    scraper: IScraper = Depends(get_scraper),
    ranker: IRanker = Depends(get_ranker),
    embedder: IEmbedder | None = Depends(get_optional_embedder),
    decision_maker: IDecisionMaker | None = Depends(get_optional_decision_maker),
    user: User = Depends(get_current_elevated_user),
    session: AsyncSession = Depends(get_session),
    source_types: list[str] = Query(default=[]),
    turn: TurnResources = Depends(get_turn_resources),
    note_store: NoteStore = Depends(get_note_store),
):
    """Run one persona turn at the chosen depth (decision 10).

    `persona_id` is required (422 by schema); a foreign persona is a
    404. The whole mode is elevated; depth is a budget choice, not a
    role choice. quick = the free loop with persona persistence;
    research/deep = the research pipeline (deep adds scout + reflect)
    with persona search constraints.
    """
    persona = await PersonaRepository(session).get_for_user(
        user.id, request.persona_id
    )
    if persona is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Persona not found."
        )
    depth = _resolve_depth(request, persona)
    ensure_source_types(request.source_types, declared_source_types)
    refs = await resolve_for_ask(
        session, user.id, None, request.attachment_ids,
        require_same_conversation=False,
    )
    ensure_vision(turn.model_capabilities, refs)
    images = await fetch_image_blobs(
        session, refs, settings.ATTACHMENT_MAX_IMAGES
    )
    att_dicts = [
        {"id": a.id, "filename": a.filename, "conversation_id": a.conversation_id}
        for a in refs
    ]

    budget = make_budget(persona.default_parameters)

    if depth == "quick":
        if request.background:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    "background is only supported for the research and "
                    "deep depths; quick turns stream."
                ),
            )
        return await _persona_quick_turn(
            request, persona, model, searching, scraper, ranker,
            decision_maker, user, session, turn, att_dicts, images, budget,
        )

    # research / deep: the shared pipeline with persona constraints.
    if request.background:
        return await _enqueue_research_turn(
            request.query, searching, model, scraper, ranker, embedder,
            session, user, request.source_types, turn, att_dicts, images,
            scout=(depth == "deep"), reflect=(depth == "deep"), persona=persona,
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
            research_sse_adapter(
                request.query,
                searching,
                model,
                scraper,
                ranker,
                embedder=embedder,
                session=session,
                source_types=request.source_types,
                attachments=att_dicts,
                images=images,
                note_store=note_store,
                scout=(depth == "deep"),
                reflect=(depth == "deep"),
                persona=persona,
                turn=turn,
            ),
            media_type="text/event-stream",
        )
    return await research_json_adapter(
        request.query,
        searching,
        model,
        scraper,
        ranker,
        embedder=embedder,
        session=session,
        source_types=request.source_types,
        attachments=att_dicts,
        images=images,
        note_store=note_store,
        scout=(depth == "deep"),
        reflect=(depth == "deep"),
        persona=persona,
        turn=turn,
    )


async def _persona_quick_turn(
    request: PersonaAskRequest,
    persona,
    model: BaseChatModel,
    searching: ISearchProvider,
    scraper: IScraper,
    ranker: IRanker,
    decision_maker: IDecisionMaker | None,
    user: User,
    session: AsyncSession,
    turn: TurnResources,
    attachments: list[dict],
    images: list[bytes] | None,
    budget,
):
    """Quick depth: the free loop over a persona-scoped conversation."""
    from src.core.ask.services.search import _sse_frame

    if settings.BUSY_POLICY == "reject":
        try:
            await acquire_gates(turn.keys, timeout=0)
        except GateBusy as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=provider_busy_detail(),
            ) from e
        release_gates(turn.keys)

    conversation, history, created = await _resolve_conversation(
        session, user.id, request.conversation_id, request.query
    )
    conversation.persona_id = persona.id
    session.add(conversation)
    await session.commit()

    toolbox = AskToolbox(
        searching, scraper, ranker, None, None,
        allowed_domains=persona.allowed_domains if persona else None,
        allowed_engines=persona.allowed_engines if persona else None,
        attachments=attachments,
    )

    async def stream():
        event_id = 0

        def frame(event: str, data: dict) -> str:
            nonlocal event_id
            event_id += 1
            return _sse_frame(event_id, event, data)

        try:
            async with turn.gated():
                yield frame(
                    "conversation",
                    {
                        "conversation_id": str(conversation.id),
                        "created": created,
                    },
                )
                parts: list[str] = []
                async for event, data in agent_logic(
                    request.query, model, toolbox, budget, persona,
                    images=images,
                ):
                    if event == "message":
                        parts.append(data["text"])
                    if event == "done":
                        await _save_turn(
                            session, conversation, request.query,
                            "".join(parts),
                        )
                    yield frame(event, data)
        except GateBusy:
            yield frame("error", provider_busy_detail())

    return StreamingResponse(stream(), media_type="text/event-stream")


async def _research_turn(
    query: str,
    stream: bool,
    searching: ISearchProvider,
    declared_source_types: list[str],
    model: BaseChatModel,
    scraper: IScraper,
    ranker: IRanker,
    embedder: IEmbedder | None,
    session: AsyncSession,
    source_types: list[str],
    turn: TurnResources,
    background: bool,
    user: User,
    attachment_ids: list[uuid.UUID],
    note_store: NoteStore,
    scout: bool,
    reflect: bool,
):
    ensure_source_types(source_types, declared_source_types)
    attachments = await resolve_for_ask(
        session, user.id, None, attachment_ids,
        require_same_conversation=False,
    )
    ensure_vision(turn.model_capabilities, attachments)
    images = await fetch_image_blobs(
        session, attachments, settings.ATTACHMENT_MAX_IMAGES
    )
    att_dicts = [
        {"id": a.id, "filename": a.filename, "conversation_id": a.conversation_id}
        for a in attachments
    ]
    if background:
        return await _enqueue_research_turn(
            query, searching, model, scraper, ranker, embedder,
            session, user, source_types, turn, att_dicts, images,
            scout=scout, reflect=reflect,
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
                attachments=att_dicts,
                images=images,
                note_store=note_store,
                scout=scout,
                reflect=reflect,
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
        attachments=att_dicts,
        images=images,
        note_store=note_store,
        scout=scout,
        reflect=reflect,
        turn=turn,
    )


@router.get(
    "/agent",
    dependencies=[
        Depends(get_current_elevated_user),
        Depends(require_scope("agent")),
    ],
    deprecated=True,
    description=(
        "Deprecated: use /ask/deep_research (the deep-research preset). "
        "This free agent loop is kept temporarily and removed at protocol "
        "v3."
    ),
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
    attachment_ids: list[uuid.UUID] = Query(default=[]),
):
    ensure_source_types(source_types, declared_source_types)
    agent_refs = await resolve_for_ask(
        session, user.id, None, attachment_ids,
        require_same_conversation=False,
    )
    ensure_vision(turn.model_capabilities, agent_refs)
    agent_images = await fetch_image_blobs(
        session, agent_refs, settings.ATTACHMENT_MAX_IMAGES
    )
    agent_attachments = [
        {"id": a.id, "filename": a.filename, "conversation_id": a.conversation_id}
        for a in agent_refs
    ]
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
        attachments=agent_attachments,
    )

    return StreamingResponse(
        agent_sse_adapter(
            goal,
            model,
            toolbox,
            make_budget(persona.default_parameters if persona else None),
            persona=persona,
            images=agent_images,
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
    attachments: list[dict],
    images: list[bytes] | None,
    scout: bool = False,
    reflect: bool = False,
) -> JSONResponse:
    """Auto-convert a busy foreground research run into a job (PR 12)."""
    job = Job(user_id=user.id, kind="research", payload={"query": query})

    async def runner() -> str:
        async with AsyncSession(engine, expire_on_commit=False) as job_session:
            job_notes = NoteStore(NoteRepository(job_session), user.id)
            result = await research_json_adapter(
                query,
                searching,
                model,
                scraper,
                ranker,
                embedder=embedder,
                session=job_session,
                source_types=source_types,
                attachments=attachments,
                images=images,
                note_store=job_notes,
                scout=scout,
                reflect=reflect,
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


