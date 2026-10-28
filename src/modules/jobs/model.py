"""Job rows persisted queue state (PR 12 / decision 19; routes in PR 13)."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime
from sqlmodel import JSON, Column, Field, SQLModel


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
