"""JobQueue wiring for the app (PR 12 / decision 19)."""

from ...tools.db import engine
from .queue import JobQueue

_queue: JobQueue | None = None


def get_job_queue() -> JobQueue:
    """Process-wide queue singleton; persists transitions via the engine."""
    global _queue
    if _queue is None:
        from sqlalchemy.ext.asyncio import AsyncSession

        _queue = JobQueue(
            session_factory=lambda: AsyncSession(engine, expire_on_commit=False),
        )
    return _queue


def reset_job_queue() -> None:
    """Test hook: drop the singleton."""
    global _queue
    _queue = None
