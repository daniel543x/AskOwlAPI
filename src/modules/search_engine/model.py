from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON
from sqlmodel import Column, Field, SQLModel

from .source_types import KNOWN_SOURCE_TYPES


class SearchProviderConfig(SQLModel, table=True):
    id: UUID | None = Field(default_factory=uuid4, primary_key=True)
    name: str = Field(index=True)
    provider_type: str = Field(index=True)
    is_active: bool = Field(default=True)
    settings: dict[str, Any] = Field(sa_column=Column(JSON), default={})
    # Source types this provider declares (PR 11 / decision 13); ask
    # bodies validate `source_types` against this list.
    supported_source_types: list[str] = Field(
        default=list(KNOWN_SOURCE_TYPES), sa_column=Column(JSON)
    )
