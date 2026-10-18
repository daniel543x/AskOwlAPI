"""Conversation history endpoints (own conversations only)."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ...tools.db import get_session
from ..identity.auth.security import get_current_user
from ..identity.user.model import User
from .model import ConversationRead, ConversationWithMessages, MessageRead
from .repository import HistoryRepository

router = APIRouter(prefix="/history", tags=["History"])


def get_repository(
    session: AsyncSession = Depends(get_session),
) -> HistoryRepository:
    return HistoryRepository(session)


@router.get("", response_model=list[ConversationRead])
async def list_conversations(
    repo: HistoryRepository = Depends(get_repository),
    user: User = Depends(get_current_user),
):
    return await repo.list_conversations(user.id)


@router.get("/{conversation_id}", response_model=ConversationWithMessages)
async def get_conversation(
    conversation_id: UUID,
    repo: HistoryRepository = Depends(get_repository),
    user: User = Depends(get_current_user),
):
    conversation = await repo.get_conversation_for_user(user.id, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    messages = await repo.get_messages(conversation.id)
    return ConversationWithMessages(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[
            MessageRead(
                id=m.id, role=m.role, content=m.content, created_at=m.created_at
            )
            for m in messages
        ],
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: UUID,
    repo: HistoryRepository = Depends(get_repository),
    user: User = Depends(get_current_user),
):
    conversation = await repo.get_conversation_for_user(user.id, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    await repo.delete_conversation(conversation)
