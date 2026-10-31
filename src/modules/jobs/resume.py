"""Rebuilding and re-enqueueing interrupted background runs (WU-17).

A restart leaves jobs `running` in the table while their in-memory
runner closures are gone. Instead of failing them (PR 13), rebuild the
runner from the job row: kind `search` replays the whole turn (cheap,
idempotent by content hashing); kind `research` continues — the note
store is seeded from the run's persisted note ids and the legs whose
task already has a note are skipped by the pipeline's own resume logic.

Tool/model resolution mirrors what the enqueue path used: the job's
user supplies roles, the profile default biases the chat model, and the
underlying factories build the toolbox pieces from the rebuilt session.
"""

import logging
import uuid

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from ...tools.db import engine
from ...tools.settings import settings
from ..identity.role.model import Role, UserRole
from ..jobs.model import Job, JobStatus
from ..profile.model import UserPreferences

logger = logging.getLogger(__name__)

INTERRUPTED_MESSAGE = "Interrupted by a server restart; re-run the turn."


async def _user_roles(session: AsyncSession, user_id: uuid.UUID) -> list[str]:
    statement = (
        select(Role.name)
        .join(UserRole, UserRole.role_id == Role.id)  # type: ignore[arg-type]
        .where(UserRole.user_id == user_id)
    )
    return [name for (name,) in (await session.exec(statement)).all()]


async def _profile_model_id(
    session: AsyncSession, user_id: uuid.UUID
) -> uuid.UUID | None:
    prefs = (
        await session.exec(
            select(UserPreferences).where(UserPreferences.user_id == user_id)
        )
    ).first()
    return prefs.default_model_id if prefs else None


async def build_runner(job: Job, session: AsyncSession):
    """A runner closure for a persisted job row, or None = not resumable."""
    if job.kind == "search":
        return _search_runner(job)
    if job.kind == "research":
        return await _research_runner(job, session)
    logger.warning(
        "No resume wiring for job kind %r (job %s).", job.kind, job.id
    )
    return None


def _search_runner(job: Job):
    payload = job.payload or {}
    query = payload.get("query") or ""
    conversation_id = payload.get("conversation_id")

    async def runner() -> str:
        from ...core.ask.services.search import search_json_adapter
        from ..owl_reranker.factory import (
            RankerFactory,
            _resolve_ranker_model,
        )
        from ..scrapy.factory import ScraperFactory
        from ..search_engine.factory import SearchProviderFactory

        async with AsyncSession(engine, expire_on_commit=False) as session:
            searching = await SearchProviderFactory.create_provider(session, None)
            scraper = await ScraperFactory.create(session)
            resolved_ranker = await _resolve_ranker_model(session, None)
            ranker = await RankerFactory.from_resolved(resolved_ranker)
            model = await _chat_model(session, job)
            result = await search_json_adapter(
                query,
                searching,
                model,
                scraper,
                ranker,
                session=session,
                user_id=job.user_id,
                conversation_id=conversation_id,
            )
        return (result.get("answer") or "")[:4000]

    return runner


async def _research_runner(job: Job, session: AsyncSession):
    payload = job.payload or {}
    research_state = payload.get("research") or {}
    query = payload.get("query") or ""
    note_ids = research_state.get("note_ids") or []
    scout = bool(payload.get("scout"))
    reflect = bool(payload.get("reflect"))

    async def runner() -> str:
        from ...core.research.services.pipeline import research_json_adapter
        from ..notes.repository import NoteRepository
        from ..notes.store import NoteStore
        from ..owl_reranker.factory import (
            RankerFactory,
            _resolve_ranker_model,
        )
        from ..personas.repository import PersonaRepository
        from ..scrapy.factory import ScraperFactory
        from ..search_engine.factory import SearchProviderFactory

        async with AsyncSession(engine, expire_on_commit=False) as job_session:
            searching = await SearchProviderFactory.create_provider(job_session, None)
            scraper = await ScraperFactory.create(job_session)
            resolved_ranker = await _resolve_ranker_model(job_session, None)
            ranker = await RankerFactory.from_resolved(resolved_ranker)
            model = await _chat_model(job_session, job)
            persona = None
            persona_id = payload.get("persona_id")
            if persona_id:
                persona = await PersonaRepository(job_session).get_for_user(
                    job.user_id, uuid.UUID(persona_id)
                )
            seed = (
                await NoteRepository(job_session).get_by_ids(note_ids)
                if note_ids
                else []
            )
            note_store = NoteStore(
                NoteRepository(job_session), job.user_id, seeded=list(seed)
            )
            result = await research_json_adapter(
                query,
                searching,
                model,
                scraper,
                ranker,
                session=job_session,
                note_store=note_store,
                persona=persona,
                scout=scout,
                reflect=reflect,
                progress=_progress_writer(job),
            )
        return (result.get("report") or "")[:4000]

    return runner


async def _chat_model(session: AsyncSession, job: Job):
    from ..llm_provider.factory import ChatModelFactory
    from ..llm_provider.model import ModelKind
    from ..llm_provider.repository import ModelRepository

    repo = ModelRepository(session)
    resolved = await repo.resolve_model(
        kind=ModelKind.CHAT.value,
        user_id=job.user_id,
        user_roles=await _user_roles(session, job.user_id),
        requested_model_id=None,
        profile_model_id=await _profile_model_id(session, job.user_id),
    )
    factory = ChatModelFactory(session)
    return factory.create_chat_model(resolved)


def _progress_writer(job: Job):
    """A hook the pipeline awaits to persist plan/note progress."""

    async def hook(data: dict) -> None:
        state = dict(job.payload.get("research") or {})
        if "planned_tasks" in data:
            state["planned_tasks"] = data["planned_tasks"]
        if "note_ids" in data:
            existing = list(state.get("note_ids") or [])
            for note_id in data["note_ids"]:
                if note_id not in existing:
                    existing.append(note_id)
            state["note_ids"] = existing
        job.payload = {**job.payload, "research": state}
        async with AsyncSession(engine, expire_on_commit=False) as session:
            await session.merge(job)
            await session.commit()

    return hook


async def requeue_orphans(
    queue,
    *,
    session_factory=None,
    jobs_provider=None,
    registry=None,
) -> int:
    """Re-enqueue jobs a restart left `running`; retry-cap aware."""
    session_factory = session_factory or (
        lambda: AsyncSession(engine, expire_on_commit=False)
    )

    async def default_registry(job: Job):
        async with session_factory() as session:
            return await build_runner(job, session)

    registry = registry or default_registry

    async def default_jobs():
        async with session_factory() as session:
            rows = (
                await session.exec(select(Job).where(Job.status == JobStatus.RUNNING))
            ).all()
            return list(rows)

    jobs_provider = jobs_provider or default_jobs

    count = 0
    for job in await jobs_provider():
        if job.retries >= settings.JOB_MAX_RETRIES:
            job.status = JobStatus.FAILED
            job.error = INTERRUPTED_MESSAGE
            async with session_factory() as session:
                await session.merge(job)
                await session.commit()
            logger.warning(
                "Interrupted job %s exceeded the resume cap; failed.", job.id
            )
            continue
        runner = registry(job)
        if runner is None:
            job.status = JobStatus.FAILED
            job.error = INTERRUPTED_MESSAGE
            async with session_factory() as session:
                await session.merge(job)
                await session.commit()
            logger.warning(
                "Interrupted job %s is not resumable; failed.", job.id
            )
            continue
        job.status = JobStatus.QUEUED
        job.retries += 1
        job.error = None
        job.started_at = None
        queue.requeue(job, runner)
        async with session_factory() as session:
            await session.merge(job)
            await session.commit()
        count += 1
        logger.info(
            "Re-enqueued interrupted %s job %s (resume %d).",
            job.kind, job.id, job.retries,
        )
    return count
