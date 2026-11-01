"""Job rows persisted queue state (PR 12 / decision 19; routes in PR 13)."""

import logging
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime
from sqlmodel import JSON, Column, Field, SQLModel

logger = logging.getLogger(__name__)


class JobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


ACTIVE_STATUSES = (JobStatus.QUEUED, JobStatus.RUNNING)


class Job(SQLModel, table=True):
    """One background ask run (research-grade modes; decision 15).

    The row IS the queue state: status transitions are persisted by the
    JobQueue so a restart can sweep orphans (PR 13) and the InBox can
    render pending/running/done.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: uuid.UUID = Field(foreign_key="user.id", index=True)
    kind: str = Field(index=True)  # search | research | agent
    status: str = Field(default=JobStatus.QUEUED, index=True)
    queue_position: int | None = Field(default=None)
    conversation_id: uuid.UUID | None = Field(default=None)
    payload: dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    result_summary: str | None = Field(default=None)
    error: str | None = Field(default=None)
    seen: bool = Field(default=False)
    retries: int = Field(default=0)
    created_at: datetime | None = Field(
        default=None, sa_type=DateTime(timezone=True)
    )
    started_at: datetime | None = Field(
        default=None, sa_type=DateTime(timezone=True)
    )
    finished_at: datetime | None = Field(
        default=None, sa_type=DateTime(timezone=True)
    )


def link_conversation(
    job: Job,
    result: dict[str, Any] | None,
    fallback: uuid.UUID | None = None,
) -> None:
    """Stamp the run's conversation onto the job row (webapp API-NOTES #10).

    The runner result's `conversation_id` (the conversation the turn was
    actually saved to) wins; `fallback` is the requested conversation for
    runs that reuse one. The queue persists the row with the succeeded
    transition, so the InBox can link to the full answer conversation.
    """
    raw = (result or {}).get("conversation_id")
    if raw:
        try:
            job.conversation_id = uuid.UUID(str(raw))
            return
        except ValueError:
            logger.warning(
                "Job %s result carried an unparsable conversation_id.",
                job.id,
            )
    job.conversation_id = fallback
