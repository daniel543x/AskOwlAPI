"""Database access for attachments (PR 15)."""

import uuid

from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from .model import Attachment, AttachmentRef


class AttachmentRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, attachment: Attachment) -> Attachment:
        self.session.add(attachment)
        await self.session.commit()
        await self.session.refresh(attachment)
        return attachment

    async def get_for_user(
        self, attachment_id: uuid.UUID, user_id: uuid.UUID
    ) -> Attachment | None:
        attachment = await self.session.get(Attachment, attachment_id)
        if attachment is None or attachment.user_id != user_id:
            return None
        return attachment

    async def list_for_conversation(
        self, conversation_id: uuid.UUID
    ) -> list[Attachment]:
        statement = (
            select(Attachment)
            .where(col(Attachment.conversation_id) == conversation_id)
            .order_by(col(Attachment.created_at))
        )
        return list((await self.session.exec(statement)).all())

    async def resolve_refs(
        self, user_id: uuid.UUID, attachment_ids: list[uuid.UUID]
    ) -> dict[uuid.UUID, Attachment]:
        """The caller's attachments among the requested ids."""
        if not attachment_ids:
            return {}
        statement = select(Attachment).where(
            col(Attachment.id).in_(attachment_ids),
            col(Attachment.user_id) == user_id,
        )
        found = list((await self.session.exec(statement)).all())
        return {a.id: a for a in found}

    async def chunks_for_attachments(
        self, attachment_ids: list[uuid.UUID], limit: int
    ) -> list[dict]:
        """Stored chunk texts for the given attachments (metadata-scoped)."""
        from ..chunk_store.repository import ChunkRepository

        return await ChunkRepository(self.session).chunks_by_attachment_ids(
            attachment_ids, limit
        )


def attachment_chunk_metadata(attachment: Attachment, sequence: int) -> dict:
    """Chunk metadata for one upload's parts (PR 15)."""
    return {
        "url": f"attachment://{attachment.id}/{sequence}",
        "title": attachment.filename,
        "attachment_id": str(attachment.id),
        "source": "attachment",
    }


def refs_of(attachments: list[Attachment]) -> list[AttachmentRef]:
    return [
        AttachmentRef(
            id=a.id, filename=a.filename, conversation_id=a.conversation_id
        )
        for a in attachments
    ]
