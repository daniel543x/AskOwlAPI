"""Attachment metadata + API schemas (PR 15 / decision 14)."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime
from sqlmodel import Field, SQLModel

# Allowed upload families; dispatch keys for the extractor.
TEXT_SUFFIXES = (".txt", ".md")
PDF_SUFFIXES = (".pdf",)
DOCX_SUFFIXES = (".docx",)
HTML_SUFFIXES = (".html", ".htm")
ALLOWED_SUFFIXES = TEXT_SUFFIXES + PDF_SUFFIXES + DOCX_SUFFIXES + HTML_SUFFIXES


class Attachment(SQLModel, table=True):
    """One uploaded text file, parsed into conversation-scoped chunks.

    The row holds metadata only — the extracted text lives in the chunk
    store (conversation-scoped, decision 21: uploads are conversation-
    scoped unconditionally, web chunks stay global).
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: uuid.UUID = Field(foreign_key="user.id", index=True)
    conversation_id: uuid.UUID = Field(
        foreign_key="conversation.id", index=True
    )
    filename: str
    content_type: str = ""
    size_bytes: int = 0
    extracted_chars: int = 0
    created_at: datetime | None = Field(
        default=None, sa_type=DateTime(timezone=True)
    )


class AttachmentRead(SQLModel):
    id: uuid.UUID
    conversation_id: uuid.UUID
    filename: str
    content_type: str
    size_bytes: int
    extracted_chars: int
    created_at: datetime | None


class AttachmentRef(SQLModel):
    """Validated attachment for a turn (agent toolbox + context injection)."""

    id: uuid.UUID
    filename: str
    conversation_id: uuid.UUID
