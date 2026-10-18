"""API-key repository: create/list/revoke + hash lookup for auth."""

import logging
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from ..identity.user.model import User
from .model import ApiKey, ApiKeyCreate, generate_api_key, validate_scopes

logger = logging.getLogger(__name__)


class ApiKeyRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, user_id: UUID, data: ApiKeyCreate) -> tuple[ApiKey, str]:
        """Create a key; returns (stored_row, raw_key_shown_once)."""
        scopes = validate_scopes(data.scopes or ["search"])
        raw, key_hash, prefix = generate_api_key()
        api_key = ApiKey(
            user_id=user_id,
            name=data.name,
            key_hash=key_hash,
            key_prefix=prefix,
            scopes=scopes,
            expires_at=data.expires_at,
        )
        self.session.add(api_key)
        await self.session.commit()
        await self.session.refresh(api_key)
        return api_key, raw

    async def list_for_user(self, user_id: UUID) -> list[ApiKey]:
        statement = (
            select(ApiKey)
            .where(ApiKey.user_id == user_id)
            .order_by(ApiKey.created_at.desc())
        )
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_for_user(self, user_id: UUID, api_key_id: UUID) -> ApiKey | None:
        api_key = await self.session.get(ApiKey, api_key_id)
        if api_key is None or api_key.user_id != user_id:
            return None
        return api_key

    async def revoke(self, api_key: ApiKey) -> ApiKey:
        api_key.is_active = False
        self.session.add(api_key)
        await self.session.commit()
        await self.session.refresh(api_key)
        return api_key

    async def resolve(self, raw_key: str) -> tuple[ApiKey, User] | None:
        """Find an active, unexpired key by its hash and load its user.

        Returns None for unknown/inactive/expired keys or inactive users;
        callers surface a generic 401 so probing learns nothing.
        """
        from .model import hash_api_key, is_expired

        statement = select(ApiKey).where(
            ApiKey.key_hash == hash_api_key(raw_key), ApiKey.is_active
        )
        result = await self.session.execute(statement)
        api_key = result.scalar_one_or_none()
        if api_key is None or is_expired(api_key):
            return None

        user = await self.session.get(User, api_key.user_id)
        if user is None or not user.is_active:
            return None

        api_key.last_used_at = datetime.now(UTC)
        self.session.add(api_key)
        await self.session.commit()
        return api_key, user
