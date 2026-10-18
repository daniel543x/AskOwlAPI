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
from sqlmodel import Field, SQLModel


class Persona(SQLModel, table=True):
    """One user-owned persona (custom search agent profile)."""

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="user.id", index=True)
    name: str = Field(max_length=100, index=True)
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
    system_context: str = ""
    allowed_domains: list[str] = Field(default_factory=list)
    allowed_engines: list[str] = Field(default_factory=list)
    allowed_keywords: list[str] = Field(default_factory=list)
    default_parameters: dict[str, Any] = Field(default_factory=dict)


class PersonaUpdate(SQLModel):
    name: str | None = None
    system_context: str | None = None
    allowed_domains: list[str] | None = None
    allowed_engines: list[str] | None = None
    allowed_keywords: list[str] | None = None
    default_parameters: dict[str, Any] | None = None


class PersonaRead(SQLModel):
    id: UUID
    name: str
    system_context: str
    allowed_domains: list[str]
    allowed_engines: list[str]
    allowed_keywords: list[str]
    default_parameters: dict[str, Any]
    created_at: datetime
    updated_at: datetime | None
