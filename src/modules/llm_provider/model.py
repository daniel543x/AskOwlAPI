import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import DateTime, PrimaryKeyConstraint
from sqlmodel import JSON, Column, Field, Relationship, SQLModel


class ModelKind(StrEnum):
    """The function a catalog entry serves. One registry, many kinds."""

    CHAT = "chat"
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    DECISION = "decision"


class ProviderHealth(StrEnum):
    UP = "up"
    DOWN = "down"
    LOCAL = "local"  # in-process implementation, nothing to probe
    UNKNOWN = "unknown"


# <----- Tables ----->


class ModelProvider(SQLModel, table=True):
    """A model endpoint: cloud, host, LAN, or in-process (e.g. flashrank).

    Every endpoint is catalog data (base URL + encrypted API key), never a
    hardcoded localhost.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(unique=True, index=True)
    provider_type: str = Field(index=True)  # 'ollama', 'openai', 'flashrank', ...
    base_url: str | None = Field(default=None)
    api_key_encrypted: str | None = Field(default=None)

    is_active: bool = Field(default=True)
    health_status: str = Field(default=ProviderHealth.UNKNOWN.value)
    last_probe_at: datetime | None = Field(
        default=None, sa_type=DateTime(timezone=True)
    )

    models: list[ModelCatalogEntry] = Relationship(back_populates="provider")


class ModelCatalogEntry(SQLModel, table=True):
    """One selectable model served by a provider."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    provider_id: uuid.UUID = Field(foreign_key="modelprovider.id", index=True)

    kind: str = Field(default=ModelKind.CHAT.value, index=True)
    model_name: str = Field()
    display_name: str = Field(default="")
    default_params: dict[str, Any] = Field(default={}, sa_column=Column(JSON))

    is_active: bool = Field(default=False)
    required_role: str = Field(default="user")
    capabilities: list[str] = Field(default=[], sa_column=Column(JSON))
    # Embeddings only: vector dimensions, so multiple embedders can coexist.
    vector_dims: int | None = Field(default=None)

    provider: ModelProvider | None = Relationship(back_populates="models")


class UserModelPreference(SQLModel, table=True):
    """Per-user default model for a kind ('chat', 'embedding', ...)."""

    __table_args__ = (PrimaryKeyConstraint("user_id", "kind"),)

    user_id: uuid.UUID = Field(foreign_key="user.id")
    kind: str = Field(index=True)
    selected_model_id: uuid.UUID = Field(foreign_key="modelcatalogentry.id")


# Key-value store for global defaults. Keys are namespaced per kind:
# "default_model_id:chat", "default_model_id:reranker", ...
class DefaultSettings(SQLModel, table=True):
    key: str = Field(primary_key=True)
    value: str = Field()


# <----- API schemas ----->


class ProviderCreate(SQLModel):
    name: str
    provider_type: str
    base_url: str | None = None
    api_key: str | None = None  # plaintext; stored encrypted only
    is_active: bool = True


class ProviderUpdate(SQLModel):
    name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    is_active: bool | None = None


class ProviderRead(SQLModel):
    id: uuid.UUID
    name: str
    provider_type: str
    base_url: str | None
    is_active: bool
    health_status: str
    last_probe_at: datetime | None
    has_api_key: bool = False


class ModelEntryCreate(SQLModel):
    provider_id: uuid.UUID
    kind: ModelKind
    model_name: str
    display_name: str = ""
    default_params: dict[str, Any] = {}
    is_active: bool = False
    required_role: str = "user"
    capabilities: list[str] = []
    vector_dims: int | None = None


class ModelEntryUpdate(SQLModel):
    model_name: str | None = None
    display_name: str | None = None
    default_params: dict[str, Any] | None = None
    is_active: bool | None = None
    required_role: str | None = None
    capabilities: list[str] | None = None
    vector_dims: int | None = None


class ModelEntryRead(SQLModel):
    id: uuid.UUID
    provider_id: uuid.UUID
    kind: str
    model_name: str
    display_name: str
    is_active: bool
    required_role: str
    capabilities: list[str]
    vector_dims: int | None


class ModelWithProviderRead(ModelEntryRead):
    provider_name: str | None = None
    provider_type: str | None = None


class PreferenceRead(SQLModel):
    user_id: uuid.UUID
    kind: str
    selected_model_id: uuid.UUID


class DefaultModelRead(SQLModel):
    kind: str
    model_id: uuid.UUID
    model_name: str | None = None
