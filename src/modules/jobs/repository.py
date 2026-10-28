"""Database access for the Inbox (PR 13): list, seen, cancel, sweep."""

import uuid
from datetime import UTC, datetime

from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession as SqlModelSession

from .model import Job, JobStatus


class JobRepository:
    def __init__(self, session: SqlModelSession):
        self.session = session

    async def list_for_user(
        self, user_id: uuid.UUID, seen: bool | None = None
    ) -> list[Job]:
        """The caller's jobs, newest first; `seen=None` = everything."""
        statement = (
            select(Job)
            .where(Job.user_id == user_id)
            .order_by(col(Job.created_at).desc(), col(Job.id).desc())
        )
        if seen is not None:
            statement = statement.where(Job.seen == seen)
        return list((await self.session.exec(statement)).all())

    async def get_for_user(
        self, job_id: uuid.UUID, user_id: uuid.UUID
    ) -> Job | None:
        job = await self.session.get(Job, job_id)
        if job is None or job.user_id != user_id:
            return None
        return job

    async def mark_seen(self, job: Job) -> Job:
        job.seen = True
        self.session.add(job)
        await self.session.commit()
        await self.session.refresh(job)
        return job

    async def mark_cancelled(self, job: Job) -> Job:
        """Record a cancellation for a job the queue no longer owns."""
        job.status = JobStatus.CANCELLED
        job.finished_at = datetime.now(UTC)
        self.session.add(job)
        await self.session.commit()
        await self.session.refresh(job)
        return job

    async def sweep_orphans(self) -> int:
        """Mark jobs left `running` by a restart as failed (PR 13).

        The in-process queue loses its in-flight turns on restart; without
        the sweep they would show as running forever.
        """
        statement = select(Job).where(Job.status == JobStatus.RUNNING)
        orphans = list((await self.session.exec(statement)).all())
        for job in orphans:
            job.status = JobStatus.FAILED
            job.error = "Interrupted by a server restart; re-run the turn."
            job.finished_at = datetime.now(UTC)
            self.session.add(job)
        if orphans:
            await self.session.commit()
        return len(orphans)
