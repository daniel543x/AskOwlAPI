from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime
from sqlmodel import Column, Field, SQLModel


class ScrapySettings(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    name: str = Field(unique=True, index=True)

    enabled: bool = Field(default=False)
    is_default: bool = Field(default=False)

    api_key: str | None = Field(default=None)
    settings: dict[str, Any] = Field(sa_column=Column(JSON), default={})


class ScrapedPage(SQLModel, table=True):
    """Per-URL scraped-content cache row (WU-10 / AR-1).

    Keyed by the sha256 of the normalized URL so tracking-param variants
    share one entry. Survives restarts; evicted implicitly by the TTL check
    at read time (rows are overwritten, never pruned eagerly).
    """

    # Explicit snake_case table name (class-name default would be "scrapedpage").
    __tablename__: str = "scraped_page"  # pyright: ignore[reportAssignmentType, reportIncompatibleVariableOverride]

    url_hash: str = Field(primary_key=True)
    url: str
    content: str
    http_status: int | None = Field(default=None)
    fetched_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_type=DateTime(timezone=True),
        index=True,
    )
