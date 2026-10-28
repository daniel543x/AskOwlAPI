"""JobQueue over the resource gates (PR 12 / decision 19).

FIFO with per-user fairness: pending jobs run round-robin across users,
and no user may exceed the per-user running cap, so several users never
starve each other. State lives in the `job` table (the row IS the queue
state). In-process asyncio is the default backend, single-node by
design; a Redis backend arrives with the multi-node topology (decision
19, same lazy pattern as ICache). External 429s are retried with backoff
honoring Retry-After when the raised error exposes it.
"""

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from ...tools.settings import settings
from .events import JobEventBus
from .model import Job, JobStatus

logger = logging.getLogger(__name__)

Runner = Callable[[], Awaitable[str]]


def _retry_after_of(exc: BaseException) -> float | None:
    """Retry-After seconds from a rate-limit error, when exposed."""
    retry_after = getattr(exc, "retry_after", None)
    if isinstance(retry_after, (int, float)) and retry_after >= 0:
        return float(retry_after)
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return None  # rate-limited, unknown delay → default backoff
    return None


class JobQueue:
    """In-process asyncio queue with per-user fairness.

    `session_factory` (optional; the app passes the engine-backed factory)
    persists every status transition; tests omit it and assert on the
    in-memory rows.
    """

    def __init__(
        self,
        per_user_running_cap: int | None = None,
        session_factory: Callable[[], AsyncSession] | None = None,
        event_bus: JobEventBus | None = None,
    ):
        self.per_user_cap = per_user_running_cap or settings.JOB_PER_USER_RUNNING_CAP
        self._session_factory = session_factory
        self.event_bus = event_bus
        self._pending: list[tuple[Job, Runner]] = []
        self._running_per_user: dict[uuid.UUID, int] = {}
        self._worker: asyncio.Task | None = None
        self._wake: asyncio.Event = asyncio.Event()

    def _notify(self, job: Job) -> None:
        if self.event_bus is None:
            return
        self.event_bus.publish(
            job.user_id,
            {
                "job_id": str(job.id),
                "kind": job.kind,
                "status": job.status,
                "queue_position": job.queue_position,
                "result_summary": job.result_summary,
                "error": job.error,
            },
        )

    async def _persist(self, job: Job) -> None:
        if self._session_factory is None:
            return
        async with self._session_factory() as session:
            await session.merge(job)
            await session.commit()

    # <----- Submission ----->

    async def submit(self, job: Job, runner: Runner) -> Job:
        """Persist + enqueue one job; returns the row with its position."""
        job.status = JobStatus.QUEUED
        job.created_at = datetime.now(UTC)
        job.queue_position = len(self._pending) + 1
        self._pending.append((job, runner))
        self._wake.set()
        await self._persist(job)
        self._notify(job)
        return job

    # <----- Worker ----->

    def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._wake.set()
            self._worker = asyncio.get_running_loop().create_task(
                self._worker_loop()
            )

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    def running_count(self, user_id) -> int:
        return self._running_per_user.get(user_id, 0)

    def _pick_next(self) -> tuple[Job, Runner] | None:
        """First pending job whose user is under the running cap (FIFO
        scan = round-robin fairness: an early job from a saturated user
        never blocks a later job from another user)."""
        for index, (job, runner) in enumerate(self._pending):
            if (
                self._running_per_user.get(job.user_id, 0) < self.per_user_cap
            ):
                self._pending.pop(index)
                self._resequence()
                return job, runner
        return None

    def _resequence(self) -> None:
        for position, (job, _) in enumerate(self._pending, start=1):
            job.queue_position = position

    async def _worker_loop(self) -> None:
        while True:
            self._wake.clear()
            nxt = self._pick_next()
            if nxt is None:
                if not self._pending:
                    await self._wake.wait()
                else:
                    # Pending exists but every user is at cap; wait for a
                    # running job to finish.
                    await asyncio.sleep(0.05)
                continue
            job, runner = nxt
            self._running_per_user[job.user_id] = (
                self._running_per_user.get(job.user_id, 0) + 1
            )
            asyncio.get_running_loop().create_task(self._run(job, runner))

    async def _run(self, job: Job, runner: Runner) -> None:
        try:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(UTC)
            job.queue_position = None
            await self._persist(job)
            self._notify(job)
            summary = await runner()
            job.status = JobStatus.SUCCEEDED
            job.result_summary = summary
            job.finished_at = datetime.now(UTC)
            await self._persist(job)
            self._notify(job)
        except Exception as exc:  # noqa: BLE001 — the queue is the boundary
            retry_after = _retry_after_of(exc)
            if (
                retry_after is not None or _is_rate_limited(exc)
            ) and job.retries < settings.JOB_MAX_RETRIES:
                job.retries += 1
                job.status = JobStatus.QUEUED
                delay = retry_after or settings.JOB_RETRY_DELAY_S
                logger.warning(
                    "Job %s hit a rate limit; retry %d/%d in %.1fs.",
                    job.id, job.retries, settings.JOB_MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)
                self._pending.insert(0, (job, runner))
                self._resequence()
                self._wake.set()
                await self._persist(job)
                self._notify(job)
            else:
                job.status = JobStatus.FAILED
                job.error = str(exc)
                job.finished_at = datetime.now(UTC)
                logger.exception("Job %s failed.", job.id)
                await self._persist(job)
                self._notify(job)
        finally:
            remaining = self._running_per_user.get(job.user_id, 1) - 1
            if remaining <= 0:
                self._running_per_user.pop(job.user_id, None)
            else:
                self._running_per_user[job.user_id] = remaining
            self._wake.set()

    # <----- Introspection (PR 13 builds routes on this) ----->

    def pending_snapshot(self) -> list[Job]:
        return list(self._pending)

    def queued_jobs(self) -> list[Job]:
        return [job for job, _ in self._pending]

    async def cancel(self, job_id: uuid.UUID) -> Job | None:
        """Cancel a still-queued job; None when it is running or unknown."""
        for index, (job, _) in enumerate(self._pending):
            if job.id == job_id:
                self._pending.pop(index)
                self._resequence()
                self._wake.set()
                job.status = JobStatus.CANCELLED
                job.finished_at = datetime.now(UTC)
                job.queue_position = None
                await self._persist(job)
                self._notify(job)
                return job
        return None


def _is_rate_limited(exc: BaseException) -> bool:
    status_code = getattr(exc, "status_code", None)
    return status_code == 429
