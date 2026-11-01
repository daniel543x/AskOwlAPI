"""Conversation history tables (per user) and API schemas."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON, Column, DateTime, ForeignKey
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
    # ON DELETE SET NULL applies to hard deletes (admin/recovery);
    # app-level persona deletion is a soft delete (WU-16) and keeps the
    # row, so conversations keep persona_id.
    persona_id: UUID | None = Field(
        default=None,
        sa_column=Column(
            ForeignKey("persona.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
    )
    # Soft delete (WU-16): non-null hides the row from every read and
    # rejects continuation; the data stays for admin-level recovery.
    deleted_at: datetime | None = Field(
        default=None, sa_type=DateTime(timezone=True)
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
    # Turn metadata (webapp API-NOTES #11): only the user message of a
    # turn carries it — the ask request's mode and its resolved chat
    # model, so the UI can restore the "mode · model" caption after a
    # reload. NULL for legacy rows and assistant messages.
    meta: dict | None = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
    )


class MessageRead(SQLModel):
    id: UUID
    role: str
    content: str
    created_at: datetime
    meta: dict | None = None


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
