"""Auth models: JWT payloads, login requests, and refresh-token sessions.

Refresh tokens (PLAN Phase 8, open decision #5):
- Opaque random tokens (`rt-` + url-safe), SHA-256 hash-only storage — the
  raw value is shown to the client exactly once and never needed again,
  same irreversibility precedent as API keys.
- One row per issued session (each login / device mints its own token, so
  one device can be revoked without the others).
- Single-use rotation: every refresh mints a child row and marks the parent
  rotated; replaying a rotated/revoked token revokes the whole chain.
"""

import hashlib
import secrets
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import TypedDict
from uuid import UUID, uuid4

from pydantic import BaseModel
from sqlalchemy import DateTime
from sqlmodel import Field, SQLModel

from ....tools.settings import settings

REFRESH_TOKEN_PREFIX = "rt-"
CLIENT_LABEL_MAX_LENGTH = 200


class TokenData(BaseModel):
    access_token: str
    token_type: str
    refresh_token: str


class JWTPayload(TypedDict):
    sub: str
    exp: int
    roles: list[str]


class LoginRequest(BaseModel):
    username_or_email: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str


class RefreshToken(SQLModel, table=True):
    """One refresh-token session (hash-only storage)."""

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="user.id", index=True)
    token_hash: str = Field(unique=True, index=True)
    parent_id: UUID | None = Field(default=None, foreign_key="refreshtoken.id")
    # Non-secret hint about the client (user agent) so sessions are tellable.
    client_label: str = Field(default="", max_length=CLIENT_LABEL_MAX_LENGTH)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_type=DateTime(timezone=True),
    )
    expires_at: datetime = Field(sa_type=DateTime(timezone=True))
    rotated_at: datetime | None = Field(default=None, sa_type=DateTime(timezone=True))
    revoked_at: datetime | None = Field(default=None, sa_type=DateTime(timezone=True))
    last_used_at: datetime | None = Field(default=None, sa_type=DateTime(timezone=True))


def generate_refresh_token() -> tuple[str, str]:
    """Return (raw_token, sha256_hash); the raw value is shown exactly once."""
    raw = f"{REFRESH_TOKEN_PREFIX}{secrets.token_urlsafe(32)}"
    return raw, hash_refresh_token(raw)


def hash_refresh_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def refresh_expiry(now: datetime | None = None) -> datetime:
    return (now or datetime.now(UTC)) + timedelta(
        days=settings.REFRESH_TOKEN_EXPIRE_DAYS
    )


def token_state(token: RefreshToken, now: datetime | None = None) -> str:
    """Classify a row: 'revoked' | 'rotated' | 'expired' | 'active'.

    Terminal states win over expiry so replaying an old rotated token is
    always treated as reuse detection, never as a benign expiry.
    """
    now = now or datetime.now(UTC)
    if token.revoked_at is not None:
        return "revoked"
    if token.rotated_at is not None:
        return "rotated"
    expires = token.expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=UTC)
    if now >= expires:
        return "expired"
    return "active"


def chain_descendants(
    rows: Iterable[RefreshToken], root: RefreshToken
) -> list[RefreshToken]:
    """The root and every descendant via parent links.

    This is the revocation set when a token is replayed: the whole
    rotation chain dies. Cycles are impossible in practice (each row has
    exactly one parent, created after it) but guarded anyway.
    """
    children_by_parent: dict[UUID, list[RefreshToken]] = {}
    for row in rows:
        if row.parent_id is not None:
            children_by_parent.setdefault(row.parent_id, []).append(row)

    chain: list[RefreshToken] = []
    seen: set[UUID] = set()
    stack = [root]
    while stack:
        current = stack.pop()
        if current.id in seen:
            continue
        seen.add(current.id)
        chain.append(current)
        stack.extend(children_by_parent.get(current.id, []))
    return chain
