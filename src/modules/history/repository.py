"""Conversation history repository, strictly scoped to the owning user."""

import logging
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from .model import ROLE_ASSISTANT, ROLE_USER, Conversation, Message

logger = logging.getLogger(__name__)

TITLE_LENGTH = 200


class HistoryRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_conversation_for_user(
        self, user_id: UUID, conversation_id: UUID
    ) -> Conversation | None:
        conversation = await self.session.get(Conversation, conversation_id)
        if conversation is None or conversation.user_id != user_id:
            return None
        return conversation

    async def create_conversation(self, user_id: UUID, title: str) -> Conversation:
        conversation = Conversation(
            user_id=user_id, title=title[:TITLE_LENGTH] or "Conversation"
        )
        self.session.add(conversation)
        await self.session.commit()
        await self.session.refresh(conversation)
        return conversation

    async def list_conversations(
        self, user_id: UUID, persona_id: UUID | None = None
    ) -> list[Conversation]:
        """The caller's conversations, newest first.

        With `persona_id` (webapp API-NOTES #4), only conversations scoped
        to that persona (NULL rows excluded).
        """
        statement = select(Conversation).where(Conversation.user_id == user_id)
        if persona_id is not None:
            statement = statement.where(Conversation.persona_id == persona_id)
        statement = statement.order_by(col(Conversation.created_at).desc())
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def get_messages(self, conversation_id: UUID) -> list[Message]:
        statement = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(col(Message.created_at), col(Message.id))
        )
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def add_turn(
        self, conversation: Conversation, user_content: str, assistant_content: str
    ) -> tuple[Message, Message]:
        """Append one user+assistant turn and bump the conversation timestamp."""
        user_message = Message(
            conversation_id=conversation.id, role=ROLE_USER, content=user_content
        )
        assistant_message = Message(
            conversation_id=conversation.id,
            role=ROLE_ASSISTANT,
            content=assistant_content,
        )
        self.session.add(user_message)
        self.session.add(assistant_message)
        conversation.updated_at = datetime.now(UTC)
        self.session.add(conversation)
        await self.session.commit()
        await self.session.refresh(user_message)
        await self.session.refresh(assistant_message)
        return user_message, assistant_message

    async def delete_conversation(self, conversation: Conversation) -> None:
        statement = select(Message).where(Message.conversation_id == conversation.id)
        for message in (await self.session.execute(statement)).scalars():
            await self.session.delete(message)
        # Flush so the child DELETEs precede the parent DELETE (FK order).
        await self.session.flush()
        await self.session.delete(conversation)
        await self.session.commit()
