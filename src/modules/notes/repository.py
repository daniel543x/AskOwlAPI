"""Database access for research notes (PR 18)."""

import uuid

from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from .model import ResearchNote, toc_entry


class NoteRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        user_id: uuid.UUID,
        task: str,
        content: str,
        sources_used: list[str],
        conversation_id: uuid.UUID | None = None,
    ) -> ResearchNote:
        sequence = await self._next_sequence(user_id)
        note = ResearchNote(
            user_id=user_id,
            conversation_id=conversation_id,
            sequence=sequence,
            task=task,
            content=content,
            sources_used=list(sources_used or []),
            created_at=None,
        )
        self.session.add(note)
        await self.session.commit()
        await self.session.refresh(note)
        return note

    async def _next_sequence(self, user_id: uuid.UUID) -> int:
        statement = (
            select(ResearchNote)
            .where(col(ResearchNote.user_id) == user_id)
            .order_by(col(ResearchNote.sequence).desc())
            .limit(1)
        )
        last = (await self.session.exec(statement)).first()
        return (last.sequence + 1) if last else 1

    async def get_by_ids(
        self, note_ids: list[uuid.UUID]
    ) -> list[ResearchNote]:
        """Notes in the given order (the run's assembly order)."""
        if not note_ids:
            return []
        found = {
            note.id: note
            for note in (
                await self.session.exec(
                    select(ResearchNote).where(
                        col(ResearchNote.id).in_(note_ids)
                    )
                )
            ).all()
        }
        return [found[nid] for nid in note_ids if nid in found]

    async def list_for_user(
        self, user_id: uuid.UUID, limit: int = 200
    ) -> list[ResearchNote]:
        statement = (
            select(ResearchNote)
            .where(col(ResearchNote.user_id) == user_id)
            .order_by(col(ResearchNote.sequence).desc())
            .limit(limit)
        )
        return list((await self.session.exec(statement)).all())


def toc_of(notes: list[ResearchNote]) -> list[dict]:
    return [toc_entry(note) for note in notes]
