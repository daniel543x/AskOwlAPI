"""Database access for per-user preference defaults."""

import uuid
from datetime import UTC, datetime

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from .model import PreferencesUpdate, UserPreferences


class PreferencesRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, user_id: uuid.UUID) -> UserPreferences | None:
        statement = select(UserPreferences).where(
            UserPreferences.user_id == user_id
        )
        return (await self.session.exec(statement)).first()

    async def save(
        self, user_id: uuid.UUID, data: PreferencesUpdate
    ) -> UserPreferences:
        """Replace the whole preference set (PUT semantics)."""
        row = await self.get(user_id)
        if row is None:
            row = UserPreferences(user_id=user_id)
            self.session.add(row)
        row.default_model_id = data.default_model_id
        row.default_mode = data.default_mode
        row.thinking = data.thinking
        row.effort = data.effort
        row.source_types = list(data.source_types or [])
        row.updated_at = datetime.now(UTC)
        await self.session.commit()
        await self.session.refresh(row)
        return row
