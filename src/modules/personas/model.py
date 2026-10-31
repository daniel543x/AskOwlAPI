"""Persona (custom agent) tables and API schemas.

A persona is a user-owned research profile: a system context that biases
the agent's queries and a set of search constraints that scope what the
tools may return. Personas are per user; the `agents` mode loads one by id
and constrains the Phase 4 agent loop with it.
"""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, Column, DateTime
from pydantic import field_validator
from sqlmodel import Field, SQLModel

DESCRIPTION_MAX_LENGTH = 300
TAGS_MAX_ITEMS = 8
TAG_MAX_LENGTH = 50


def _clean_tags(tags: list[str] | None) -> list[str] | None:
    """Trim, drop empties, then enforce the caps as rejections."""
    if tags is None:
        return None
    cleaned = [tag.strip() for tag in tags if tag.strip()]
    if len(cleaned) > TAGS_MAX_ITEMS:
        raise ValueError(f"At most {TAGS_MAX_ITEMS} tags are allowed.")
    if any(len(tag) > TAG_MAX_LENGTH for tag in cleaned):
        raise ValueError(f"Each tag is limited to {TAG_MAX_LENGTH} characters.")
    return cleaned


class Persona(SQLModel, table=True):
    """One user-owned persona (custom search agent profile)."""

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="user.id", index=True)
    name: str = Field(max_length=100, index=True)
    # Card + editor copy (webapp API-NOTES #1); NULL = never written.
    description: str | None = Field(default=None, max_length=DESCRIPTION_MAX_LENGTH)
    # Card chips (webapp API-NOTES #2): UI-owned vocabulary, no registry.
    tags: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    # The persona's chat model (webapp API-NOTES #5): a modelcatalogentry
    # id, validated at write time. Deliberately NO FK — a catalog deletion
    # must leave the persona intact and surface as "missing" at ask time.
    model: UUID | None = Field(default=None)
    # Free-form system context (topic, tech stack, tone) injected into the
    # agent's system prompt to bias query rewriting.
    system_context: str = Field(default="")
    # Search constraints. Empty list = unconstrained.
    allowed_domains: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    allowed_engines: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    allowed_keywords: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    # Bounded set of runtime defaults applied when this persona is used,
    # e.g. {"max_iterations": 6}. Unknown keys are ignored.
    default_parameters: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON)
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_type=DateTime(timezone=True),
    )
    updated_at: datetime | None = Field(
        default=None, sa_type=DateTime(timezone=True)
    )


class PersonaCreate(SQLModel):
    name: str = Field(max_length=100)
    description: str | None = Field(default=None, max_length=DESCRIPTION_MAX_LENGTH)
    tags: list[str] = Field(default_factory=list)
    model: UUID | None = None
    system_context: str = ""
    allowed_domains: list[str] = Field(default_factory=list)
    allowed_engines: list[str] = Field(default_factory=list)
    allowed_keywords: list[str] = Field(default_factory=list)
    default_parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tags")
    @classmethod
    def _validate_tags(cls, value: list[str]) -> list[str]:
        cleaned = _clean_tags(value)
        assert cleaned is not None
        return cleaned


class PersonaUpdate(SQLModel):
    name: str | None = None
    description: str | None = Field(default=None, max_length=DESCRIPTION_MAX_LENGTH)
    tags: list[str] | None = None
    model: UUID | None = None
    system_context: str | None = None
    allowed_domains: list[str] | None = None
    allowed_engines: list[str] | None = None
    allowed_keywords: list[str] | None = None
    default_parameters: dict[str, Any] | None = None

    @field_validator("tags")
    @classmethod
    def _validate_tags(cls, value: list[str] | None) -> list[str] | None:
        return _clean_tags(value)


class PersonaRead(SQLModel):
    id: UUID
    name: str
    description: str | None
    tags: list[str]
    model: UUID | None
    system_context: str
    allowed_domains: list[str]
    allowed_engines: list[str]
    allowed_keywords: list[str]
    default_parameters: dict[str, Any]
    created_at: datetime
    updated_at: datetime | None
