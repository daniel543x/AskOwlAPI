"""Persistent chunk store table (pgvector) for hybrid retrieval."""

import hashlib
import uuid
from datetime import UTC, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column
from sqlmodel import JSON, Field, SQLModel


class Chunk(SQLModel, table=True):
    """One embedded content chunk scraped from a page."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    content: str = Field()
    # `metadata` is reserved by SQLAlchemy declarative; python attr `meta`
    # maps to the `metadata` column (url, header hierarchy, ...).
    meta: dict[str, Any] = Field(default={}, sa_column=Column("metadata", JSON))
    # Untyped vector column: multiple embedders with different dims coexist;
    # dimensions are recorded per catalog entry.
    embedding: list[float] | None = Field(default=None, sa_column=Column(Vector()))
    source_url: str = Field(index=True)
    content_hash: str = Field(index=True, unique=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), index=True
    )


def chunk_hash(source_url: str, content: str) -> str:
    return hashlib.sha256(f"{source_url}\n{content}".encode()).hexdigest()
