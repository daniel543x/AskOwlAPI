"""Conversation history tables (per user) and API schemas."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import Column, DateTime, ForeignKey
from sqlmodel import Field, SQLModel

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
MESSAGE_ROLES = (ROLE_USER, ROLE_ASSISTANT)


class Conversation(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="user.id", index=True)
    title: str = Field(max_length=200, default="")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_type=DateTime(timezone=True),
    )
    updated_at: datetime | None = Field(
        default=None, sa_type=DateTime(timezone=True)
    )
    # Persona scope (PR 20): set when the conversation belongs to a
    # persona mode turn; NULL = an ordinary conversation.
    # ON DELETE SET NULL keeps persona conversations as ordinary history
    # when their persona is deleted (webapp API-NOTES #8).
    persona_id: UUID | None = Field(
        default=None,
        sa_column=Column(
            ForeignKey("persona.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
    )


class Message(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    conversation_id: UUID = Field(foreign_key="conversation.id", index=True)
    role: str = Field(max_length=16)
    content: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_type=DateTime(timezone=True),
    )


class MessageRead(SQLModel):
    id: UUID
    role: str
    content: str
    created_at: datetime


class ConversationRead(SQLModel):
    id: UUID
    title: str
    # Persona scope (webapp API-NOTES #4): the UI routes non-null rows
    # into /personas/{persona_id}; NULL = ordinary chat.
    persona_id: UUID | None
    created_at: datetime
    updated_at: datetime | None


class ConversationWithMessages(ConversationRead):
    messages: list[MessageRead]
