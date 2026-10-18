"""API keys for backend/client authentication.

Design (docs/architecture/api_keys.md, open decision #3):
- Keys are server-generated (`ak-` + 43 url-safe chars) and shown exactly
  once at creation; only a SHA-256 hash is stored. Hash-only (not Fernet)
  because the raw key is never needed again after issuance — unlike
  provider keys, which CryptoService keeps reversibly encrypted.
- Scopes gate modes: {search, research, agent}. JWT sessions bypass scope
  checks (role gates still apply).
- Rate limiting is per key, in-process, fixed window (single-node deploys).
"""

import hashlib
import secrets
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, Column, DateTime
from sqlmodel import Field, SQLModel

KEY_PREFIX = "ak-"


def generate_api_key() -> tuple[str, str, str]:
    """Return (raw_key, sha256_hash, display_prefix)."""
    raw = f"{KEY_PREFIX}{secrets.token_urlsafe(32)}"
    return raw, hash_api_key(raw), raw[: len(KEY_PREFIX) + 5]


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


VALID_SCOPES = ("search", "research", "agent")


class ApiKey(SQLModel, table=True):
    """One user-owned API key (hash-only storage)."""

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="user.id", index=True)
    name: str = Field(max_length=100)
    key_hash: str = Field(unique=True, index=True)
    # Short non-secret fragment so users can tell keys apart.
    key_prefix: str = Field(max_length=12)
    scopes: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    is_active: bool = Field(default=True)
    expires_at: datetime | None = Field(default=None, sa_type=DateTime(timezone=True))
    last_used_at: datetime | None = Field(
        default=None, sa_type=DateTime(timezone=True)
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_type=DateTime(timezone=True),
    )


class ApiKeyCreate(SQLModel):
    name: str = Field(max_length=100)
    scopes: list[str] = Field(default_factory=lambda: ["search"])
    expires_at: datetime | None = None


class ApiKeyRead(SQLModel):
    id: UUID
    name: str
    key_prefix: str
    scopes: list[str]
    is_active: bool
    expires_at: datetime | None
    last_used_at: datetime | None
    created_at: datetime


class ApiKeyCreated(ApiKeyRead):
    """Response of creation: includes the raw key exactly once."""

    raw_key: str


def validate_scopes(scopes: list[str]) -> list[str]:
    """Deduplicate and validate; raises ValueError on unknown scopes."""
    unique = list(dict.fromkeys(scopes))
    unknown = [scope for scope in unique if scope not in VALID_SCOPES]
    if unknown:
        raise ValueError(
            f"Unknown scopes {unknown}; valid scopes: {list(VALID_SCOPES)}."
        )
    return unique


def is_expired(api_key: ApiKey, now: datetime | None = None) -> bool:
    if api_key.expires_at is None:
        return False
    now = now or datetime.now(UTC)
    expires = api_key.expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=UTC)
    return now >= expires


def public_fields(api_key: ApiKey) -> dict[str, Any]:
    """ApiKeyRead payload (never the hash)."""
    return ApiKeyRead.model_validate(api_key, from_attributes=True).model_dump()
