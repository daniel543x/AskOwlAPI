"""Attachment metadata + API schemas (PR 15 / decision 14)."""

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, LargeBinary
from sqlmodel import Field, SQLModel

# Allowed upload families; dispatch keys for the extractor.
TEXT_SUFFIXES = (".txt", ".md")
PDF_SUFFIXES = (".pdf",)
DOCX_SUFFIXES = (".docx",)
HTML_SUFFIXES = (".html", ".htm")
ALLOWED_SUFFIXES = TEXT_SUFFIXES + PDF_SUFFIXES + DOCX_SUFFIXES + HTML_SUFFIXES


ATTACHMENT_KIND_TEXT = "text"
ATTACHMENT_KIND_IMAGE = "image"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif")


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
    # "text" (parsed into chunks) or "image" (blob + vision synthesis).
    kind: str = Field(default=ATTACHMENT_KIND_TEXT)
    size_bytes: int = 0
    extracted_chars: int = 0
    created_at: datetime | None = Field(
        default=None, sa_type=DateTime(timezone=True)
    )


class AttachmentBlob(SQLModel, table=True):
    """Raw bytes of one image attachment (PR 16); TTL-swept when orphaned.

    The attachment row holds metadata; the blob table holds the bytes so
    listing stays cheap and cleanup never touches text attachments.
    """

    attachment_id: uuid.UUID = Field(
        primary_key=True, foreign_key="attachment.id"
    )
    data: bytes = Field(sa_column=Column("data", LargeBinary))


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
    kind: str = ATTACHMENT_KIND_TEXT
