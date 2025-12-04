from typing import Any, Dict
from uuid import UUID, uuid4

from sqlalchemy import JSON
from sqlmodel import Column, Field, SQLModel


class SearchProviderConfig(SQLModel, table=True):
    id: UUID | None = Field(default_factory=uuid4, primary_key=True)
    name: str = Field(index=True)
    provider_type: str = Field(index=True)
    is_active: bool = Field(default=True)
    settings: Dict[str, Any] = Field(sa_column=Column(JSON), default={})
