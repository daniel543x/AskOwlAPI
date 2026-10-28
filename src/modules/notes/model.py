"""Research notes (PR 18 / decision 9): the DB is the source of truth."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime
from sqlmodel import JSON, Column, Field, SQLModel


class ResearchNote(SQLModel, table=True):
    """One note a research-grade turn wrote into the database.

    Notes decouple working-set growth from the synthesis context and
    preserve citations across regeneration: report assembly reads the
    rows, never the transient gather blob.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: uuid.UUID = Field(foreign_key="user.id", index=True)
    conversation_id: uuid.UUID | None = Field(
        default=None, foreign_key="conversation.id", index=True
    )
    sequence: int = Field(default=0)
    task: str
    content: str
    sources_used: list[str] = Field(default=[], sa_column=Column(JSON))
    created_at: datetime | None = Field(
        default=None, sa_type=DateTime(timezone=True)
    )


def toc_entry(note: ResearchNote) -> dict:
    """The TOC entry every note_write returns (deterministic summary)."""
    one_line = " ".join(note.content.split())[:120]
    return {
        "idx": note.sequence,
        "task": note.task,
        "one_line_summary": one_line,
        "sources_count": len(note.sources_used or []),
    }
