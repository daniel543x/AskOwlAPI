"""Refresh-token repository: issue, rotate, revoke, purge (PLAN Phase 8)."""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID

from sqlalchemy import CursorResult, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from .model import (
    RefreshToken,
    chain_descendants,
    generate_refresh_token,
    hash_refresh_token,
    refresh_expiry,
)

logger = logging.getLogger(__name__)


class RefreshTokenRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        user_id: UUID,
        client_label: str = "",
        now: datetime | None = None,
    ) -> tuple[RefreshToken, str]:
        """Start a fresh session; returns (row, raw_token_shown_once)."""
        raw, token_hash = generate_refresh_token()
        token = RefreshToken(
            user_id=user_id,
            token_hash=token_hash,
            client_label=client_label,
            expires_at=refresh_expiry(now),
        )
        self.session.add(token)
        await self.session.commit()
        await self.session.refresh(token)
        return token, raw

    async def get_by_raw(self, raw: str) -> RefreshToken | None:
        statement = select(RefreshToken).where(
            RefreshToken.token_hash == hash_refresh_token(raw)
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def rotate(
        self,
        token: RefreshToken,
        client_label: str = "",
        now: datetime | None = None,
    ) -> tuple[RefreshToken, str]:
        """Single-use rotation: mark the parent, mint a linked child.

        One commit for both writes so a crash never leaves a rotated
        parent without its child (the client would be locked out).
        """
        now = now or datetime.now(UTC)
        raw, token_hash = generate_refresh_token()
        child = RefreshToken(
            user_id=token.user_id,
            token_hash=token_hash,
            parent_id=token.id,
            client_label=client_label or token.client_label,
            expires_at=refresh_expiry(now),
        )
        token.rotated_at = now
        token.last_used_at = now
        self.session.add_all([token, child])
        await self.session.commit()
        await self.session.refresh(child)
        return child, raw

    async def revoke_chain(
        self, token: RefreshToken, now: datetime | None = None
    ) -> int:
        """Revoke the token and every descendant (stolen-token response)."""
        now = now or datetime.now(UTC)
        statement = select(RefreshToken).where(RefreshToken.user_id == token.user_id)
        result = await self.session.execute(statement)
        rows = list(result.scalars().all())

        revoked = 0
        for row in chain_descendants(rows, token):
            if row.revoked_at is None:
                row.revoked_at = now
                self.session.add(row)
                revoked += 1
        if revoked:
            await self.session.commit()
        return revoked

    async def revoke_all_for_user(
        self, user_id: UUID, now: datetime | None = None
    ) -> int:
        """Revoke every live session (password change, admin block)."""
        now = now or datetime.now(UTC)
        statement = select(RefreshToken).where(
            RefreshToken.user_id == user_id,
            col(RefreshToken.revoked_at).is_(None),
        )
        result = await self.session.execute(statement)
        rows = list(result.scalars().all())
        for row in rows:
            row.revoked_at = now
            self.session.add(row)
        if rows:
            await self.session.commit()
        return len(rows)

    async def delete_all_for_user(self, user_id: UUID) -> int:
        """Hard-delete every row (only used when the user is deleted)."""
        statement = delete(RefreshToken).where(col(RefreshToken.user_id) == user_id)
        result = await self.session.execute(statement)
        await self.session.commit()
        return cast("CursorResult[Any]", result).rowcount or 0

    async def purge_expired(
        self, retention_days: int, now: datetime | None = None
    ) -> int:
        """Delete rows past the retention window (keeps the table small).

        A row becomes purgeable once it stopped being usable (expired,
        rotated, or revoked) and its retention window has passed; until
        then it stays as reuse-detection evidence and audit history.
        """
        now = now or datetime.now(UTC)
        cutoff = now - timedelta(days=retention_days)
        statement = delete(RefreshToken).where(
            (col(RefreshToken.expires_at) < cutoff)
            | (col(RefreshToken.rotated_at) < cutoff)
            | (col(RefreshToken.revoked_at) < cutoff)
        )
        result = await self.session.execute(statement)
        await self.session.commit()
        return cast("CursorResult[Any]", result).rowcount or 0
