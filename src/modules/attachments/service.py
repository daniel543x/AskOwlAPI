"""Attachment ingestion and ask-time resolution (PR 15 / decision 14).

Uploads are parsed server-side into conversation-scoped chunks
(decision 21: uploads are conversation-scoped unconditionally; web
chunks stay global). Parsing is CPU-bound sync — everything runs
through `asyncio.to_thread`.
"""

import asyncio
import logging
import uuid
from datetime import UTC, datetime

from fastapi import HTTPException, status
from sqlmodel.ext.asyncio.session import AsyncSession

from ...core.tools.chunking import chunk_markdown_data
from ...tools.settings import settings
from ..chunk_store.repository import ChunkRepository
from .extract import extract_text
from .model import (
    ATTACHMENT_KIND_IMAGE,
    Attachment,
    AttachmentRef,
)
from .repository import AttachmentRepository, attachment_chunk_metadata

logger = logging.getLogger(__name__)


async def ingest_upload(
    session: AsyncSession,
    embedder,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    filename: str,
    content_type: str,
    data: bytes,
) -> Attachment:
    """Parse one upload, chunk + embed + store it, record the row."""
    text = await asyncio.to_thread(extract_text, filename, data)
    chunks = chunk_markdown_data(
        [{"url": f"attachment://{filename}", "content": text}]
    )

    embeddings: list[list[float] | None]
    if embedder is not None:
        embedded = await embedder.embed_texts(
            [c.get("content", "") for c in chunks]
        )
        embeddings = list(embedded)
    else:
        # No embedder configured: the chunks stay retrievable through the
        # non-vector paths (context injection, metadata queries).
        embeddings = [None] * len(chunks)

    attachment = Attachment(
        user_id=user_id,
        conversation_id=conversation_id,
        filename=filename,
        content_type=content_type,
        size_bytes=len(data),
        extracted_chars=len(text),
        created_at=datetime.now(UTC),
    )
    attachment = await AttachmentRepository(session).create(attachment)

    for sequence, chunk in enumerate(chunks):
        metadata = dict(chunk.get("metadata") or {})
        metadata.update(attachment_chunk_metadata(attachment, sequence))
        chunk["metadata"] = metadata
        chunk.setdefault("url", metadata["url"])

    await ChunkRepository(session).store_chunks(
        chunks, embeddings, conversation_id=conversation_id
    )
    logger.info(
        "Stored attachment %s (%d chunks, %d chars).",
        attachment.filename, len(chunks), len(text),
    )
    return attachment


async def resolve_for_ask(
    session: AsyncSession,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID | None,
    attachment_ids: list[uuid.UUID],
    require_same_conversation: bool = True,
) -> list[AttachmentRef]:
    """Validate `attachment_ids` for one ask turn (422 on every mismatch).

    Attachments are conversation-scoped (decision 21). The deterministic
    search mode must continue the conversation the files were uploaded
    to (`require_same_conversation`); agent/research turns reference
    files directly and retrieve through the attachments' own
    conversation scope.
    """
    if not attachment_ids:
        return []
    if conversation_id is None and require_same_conversation:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "attachment_ids require a conversation_id: continue the "
                "conversation the files were uploaded to."
            ),
        )
    found = await AttachmentRepository(session).resolve_refs(
        user_id, attachment_ids
    )
    missing = [str(a) for a in attachment_ids if a not in found]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Unknown attachment_ids: {missing}.",
        )
    if require_same_conversation:
        foreign = [
            str(a)
            for a, att in found.items()
            if att.conversation_id != conversation_id
        ]
        if foreign:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    f"Attachments {foreign} belong to a different "
                    "conversation."
                ),
            )
    return [
        AttachmentRef(
            id=a.id,
            filename=a.filename,
            conversation_id=a.conversation_id,
            kind=a.kind,
        )
        for a in found.values()
    ]


def ensure_vision(
    model_capabilities: list[str], refs: list[AttachmentRef]
) -> None:
    """Coded 422 when image attachments meet a non-vision model (decision 14).

    Never a silent drop: an image ask with a model lacking `vision`
    fails loudly; the iron rule additionally requires `tools`.
    """
    if "vision" in model_capabilities:
        return
    images = [ref.filename for ref in refs if ref.kind == ATTACHMENT_KIND_IMAGE]
    if images:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "model_lacks_vision",
                "message": (
                    "This turn includes image attachments "
                    f"({images}) but the selected model does not "
                    "support vision. Pick a vision-capable model."
                ),
            },
        )


async def fetch_image_blobs(
    session: AsyncSession, refs: list[AttachmentRef], max_images: int
) -> list[bytes]:
    """Raw bytes of the referenced image attachments, upload order."""
    from .repository import AttachmentRepository

    image_ids = [
        ref.id for ref in refs if ref.kind == ATTACHMENT_KIND_IMAGE
    ]
    if not image_ids:
        return []
    return await AttachmentRepository(session).get_image_blobs(
        image_ids, max_images
    )


async def attachment_context(
    session: AsyncSession,
    refs: list[AttachmentRef],
    max_chars: int | None = None,
) -> str:
    """Flat text block for the deterministic pipeline's context."""
    if not refs:
        return ""
    limit = max_chars or settings.ATTACHMENT_CONTEXT_MAX_CHARS
    ids = [ref["id"] if isinstance(ref, dict) else ref.id for ref in refs]
    chunks = await AttachmentRepository(session).chunks_for_attachments(
        ids, settings.ATTACHMENT_SEARCH_K * 4
    )
    parts: list[str] = []
    used = 0
    for chunk in chunks:
        title = (chunk.get("metadata") or {}).get("title", "attachment")
        part = f"[File: {title}]\n{chunk['content']}"
        if parts and used + len(part) > limit:
            break
        if not parts and len(part) > limit:
            part = part[:limit]  # always surface at least one file
        parts.append(part)
        used += len(part)
    return "\n\n".join(parts)


async def ingest_image(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    filename: str,
    content_type: str,
    data: bytes,
) -> Attachment:
    """Store one image attachment (blob only; never the search path).

    Images reach the synthesis LLM as content blocks when the model has
    the `vision` capability (decision 14) — chunks and embeddings never
    happen for images.
    """
    attachment = Attachment(
        user_id=user_id,
        conversation_id=conversation_id,
        filename=filename,
        content_type=content_type,
        size_bytes=len(data),
        extracted_chars=0,
        kind=ATTACHMENT_KIND_IMAGE,
        created_at=datetime.now(UTC),
    )
    attachment = await AttachmentRepository(session).create(attachment)
    await AttachmentRepository(session).store_image_blob(attachment.id, data)
    logger.info("Stored image attachment %s (%d bytes).", filename, len(data))
    return attachment
