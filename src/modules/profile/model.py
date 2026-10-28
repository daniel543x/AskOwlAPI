"""Server-side per-user preference defaults (PLAN Phase 10, PR 9/decision 12).

The profile is the client's initialization source, not request state: ask
requests stay explicit and stateless. Stored values are clamped against
the catalog on save (decision 12: switching the model resets
thinking/effort to that model's defaults), and the resolved view is what
the UI renders.
"""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import field_validator
from sqlalchemy import DateTime
from sqlmodel import JSON, Column, Field, SQLModel

# Modes gate routes (decision 5); `deep_research` ships with Phase 11 and
# `persona` with PR 20, but the profile may store them ahead of time.
KNOWN_MODES = ("search", "research", "deep_research", "persona")
# v1 source-type set (decision 13); PR 11 makes validation provider-aware.
KNOWN_SOURCE_TYPES = ("web", "news", "academic", "social")


class UserPreferences(SQLModel, table=True):
    """One preference set per user (the whole row is the profile)."""

    user_id: uuid.UUID = Field(primary_key=True, foreign_key="user.id")
    default_model_id: uuid.UUID | None = Field(
        default=None, foreign_key="modelcatalogentry.id"
    )
    default_mode: str | None = Field(default=None)
    thinking: bool | None = Field(default=None)
    effort: str | None = Field(default=None)
    source_types: list[str] = Field(default=[], sa_column=Column(JSON))
    updated_at: datetime | None = Field(
        default=None, sa_type=DateTime(timezone=True)
    )


class PreferencesUpdate(SQLModel):
    """Body of PUT /profile/preferences. PUT replaces the whole set."""

    default_model_id: uuid.UUID | None = None
    default_mode: Literal["search", "research", "deep_research", "persona"] | None = (
        None
    )
    thinking: bool | None = None
    effort: Literal["low", "medium", "high", "max"] | None = None
    source_types: list[str] = []

    @field_validator("source_types")
    @classmethod
    def _dedupe_source_types(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


class PreferencesView(SQLModel):
    """Stored + resolved views with the clamps that were applied."""

    stored: PreferencesStored
    resolved: PreferencesResolved
    warnings: list[str] = []


class PreferencesStored(SQLModel):
    default_model_id: uuid.UUID | None
    default_mode: str | None
    thinking: bool | None
    effort: str | None
    source_types: list[str]
    updated_at: datetime | None


class PreferencesResolved(SQLModel):
    """Stored values clamped against the effective model (decision 12)."""

    default_model_id: uuid.UUID | None
    default_mode: str | None
    thinking: bool | None
    effort: str | None
    source_types: list[str]
