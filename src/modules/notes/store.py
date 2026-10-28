"""Run-scoped note writer the toolbox tool and pipeline share (PR 18)."""

import uuid

from .model import ResearchNote, toc_entry
from .repository import NoteRepository


class NoteStore:
    """Accumulates one run's notes; the DB stays the source of truth."""

    def __init__(
        self,
        repo: NoteRepository,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID | None = None,
    ):
        self.repo = repo
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.notes: list[ResearchNote] = []

    async def write(
        self, task: str, content: str, sources_used: list[str]
    ) -> dict:
        """Persist one note; returns its TOC entry (+ id for assembly)."""
        note = await self.repo.create(
            self.user_id, task, content, sources_used, self.conversation_id
        )
        self.notes.append(note)
        entry = toc_entry(note)
        entry["id"] = str(note.id)
        return entry

    async def ordered_notes(self) -> list[ResearchNote]:
        """The run's notes in creation (sequence) order."""
        return await self.repo.get_by_ids([n.id for n in self.notes])
