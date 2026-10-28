"""JobQueue wiring for the app (PR 12 / decision 19)."""

from ...tools.db import engine
from .events import JobEventBus
from .queue import JobQueue

_queue: JobQueue | None = None
_event_bus: JobEventBus | None = None


def get_event_bus() -> JobEventBus:
    global _event_bus
    if _event_bus is None:
        _event_bus = JobEventBus()
    return _event_bus


def get_job_queue() -> JobQueue:
    """Process-wide queue singleton; persists transitions via the engine."""
    global _queue
    if _queue is None:
        from sqlalchemy.ext.asyncio import AsyncSession

        _queue = JobQueue(
            session_factory=lambda: AsyncSession(engine, expire_on_commit=False),
            event_bus=get_event_bus(),
        )
    return _queue


def reset_job_queue() -> None:
    """Test hook: drop the singletons."""
    global _queue, _event_bus
    _queue = None
    _event_bus = None
