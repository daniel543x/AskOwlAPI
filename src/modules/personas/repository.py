"""Persona repository: CRUD strictly scoped to the owning user."""

import logging
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from .model import Persona, PersonaUpdate

logger = logging.getLogger(__name__)


class PersonaRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, persona: Persona) -> Persona:
        self.session.add(persona)
        await self.session.commit()
        await self.session.refresh(persona)
        return persona

    async def list_for_user(self, user_id: UUID) -> list[Persona]:
        statement = select(Persona).where(Persona.user_id == user_id)
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_for_user(self, user_id: UUID, persona_id: UUID) -> Persona | None:
        persona = await self.session.get(Persona, persona_id)
        if persona is None or persona.user_id != user_id:
            # Other users' personas are indistinguishable from missing ones.
            return None
        return persona

    async def update(
        self, persona: Persona, changes: PersonaUpdate
    ) -> Persona:
        # exclude_unset only: an explicit null clears a nullable field, an
        # omitted field means "no change" (webapp API-NOTES #1).
        data = changes.model_dump(exclude_unset=True)
        for key, value in data.items():
            setattr(persona, key, value)
        persona.updated_at = datetime.now(UTC)
        self.session.add(persona)
        await self.session.commit()
        await self.session.refresh(persona)
        return persona

    async def delete(self, persona: Persona) -> None:
        await self.session.delete(persona)
        await self.session.commit()
