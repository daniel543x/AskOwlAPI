from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from sqlalchemy import JSON
from sqlmodel import Column, Field, SQLModel


class ScrapySettings(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    name: str = Field(unique=True, index=True)

    enabled: bool = Field(default=False)
    is_default: bool = Field(default=False)

    api_key: Optional[str] = Field(default=None)
    settings: Dict[str, Any] = Field(sa_column=Column(JSON), default={})
