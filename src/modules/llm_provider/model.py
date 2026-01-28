import uuid
from typing import List, Optional

from sqlmodel import Field, Relationship, SQLModel


# --- Provider ---
class LLMProvider(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(unique=True, index=True)

    provider_type: str = Field(index=True)
    api_key_encrypted: Optional[str] = None
    base_url: Optional[str] = None
    extra_connection_params: Optional[str] = None

    models: List["LLMConfig"] = Relationship(back_populates="provider")


# --- LLM ---
class LLMConfig(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    config_name: str = Field(unique=True, index=True)
    provider_id: uuid.UUID = Field(foreign_key="llmprovider.id")
    provider: LLMProvider = Relationship(back_populates="models")

    model_name: str

    temperature: float = Field(default=0.7)
    max_tokens: Optional[int] = None

    is_default: bool = Field(default=False)
    is_active: bool = Field(default=True)
    extra_model_params: Optional[str] = None
