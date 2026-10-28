"""InBox routes: the user's view of the background job queue (PR 13).

Pending jobs list with positions, running/done/failed notifications with
a seen flag, cancellation for still-queued jobs, and a per-user SSE
stream of `job_update` events (protocol v2.11).
"""

import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import SQLModel

from ...tools.db import get_session
from ..identity.auth.security import get_current_user
from ..identity.user.model import User
from .events import JobEventBus
from .factory import get_event_bus, get_job_queue
from .model import Job
from .repository import JobRepository


def get_repository(session: AsyncSession = Depends(get_session)):
    return JobRepository(session)


router = APIRouter(tags=["Jobs"])


class JobRead(SQLModel):
    id: uuid.UUID
    kind: str
    status: str
    queue_position: int | None
    conversation_id: uuid.UUID | None
    result_summary: str | None
    error: str | None
    seen: bool
    created_at: Any | None = None
    started_at: Any | None = None
    finished_at: Any | None = None


class JobSeenUpdate(SQLModel):
    seen: bool


def _read(job: Job) -> JobRead:
    return JobRead(
        id=job.id,
        kind=job.kind,
        status=job.status,
        queue_position=job.queue_position,
        conversation_id=job.conversation_id,
        result_summary=job.result_summary,
        error=job.error,
        seen=job.seen,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.get("/jobs", response_model=list[JobRead])
async def list_jobs(
    seen: bool | None = Query(default=None),
    repo: JobRepository = Depends(get_repository),
    user: User = Depends(get_current_user),
):
    """The caller's jobs, newest first; `?seen=false` = the Inbox badge."""
    return [_read(job) for job in await repo.list_for_user(user.id, seen)]


@router.post("/jobs/{job_id}/seen", response_model=JobRead)
async def mark_job_seen(
    job_id: uuid.UUID,
    repo: JobRepository = Depends(get_repository),
    user: User = Depends(get_current_user),
):
    job = await repo.get_for_user(job_id, user.id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Job not found."
        )
    return _read(await repo.mark_seen(job))


@router.post("/jobs/{job_id}/cancel", response_model=JobRead)
async def cancel_job(
    job_id: uuid.UUID,
    repo: JobRepository = Depends(get_repository),
    user: User = Depends(get_current_user),
):
    """Cancel a still-queued job; running jobs answer 409 (PR 13)."""
    job = await repo.get_for_user(job_id, user.id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Job not found."
        )
    cancelled = await get_job_queue().cancel(job_id)
    if cancelled is None:
        if job.status in ("running", "succeeded", "failed", "cancelled"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Job is already {job.status}; only queued jobs cancel.",
            )
        # In the queue's memory the job is gone (e.g. restart lost it);
        # record the cancellation on the row directly.
        return _read(await repo.mark_cancelled(job))
    return _read(cancelled)


@router.get("/jobs/events")
async def job_events(
    user: User = Depends(get_current_user),
    bus: JobEventBus = Depends(get_event_bus),
):
    """Per-user SSE stream of `job_update` events (protocol v2.11)."""
    user_id = user.id

    async def stream():
        listener = bus.subscribe(user_id)
        try:
            while True:
                payload = await listener.get()
                yield (
                    "event: job_update\n"
                    f"data: {json.dumps(payload, default=str)}\n\n"
                )
        finally:
            bus.unsubscribe(user_id, listener)

    return StreamingResponse(stream(), media_type="text/event-stream")
