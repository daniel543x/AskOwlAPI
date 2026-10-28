"""Attachment upload route (PR 15 / decision 14)."""

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlmodel.ext.asyncio.session import AsyncSession

from ...tools.db import get_session
from ...tools.settings import settings
from ..embedding.factory import get_optional_embedder
from ..history.repository import HistoryRepository
from ..identity.auth.security import get_current_user
from ..identity.user.model import User
from .model import (
    ALLOWED_SUFFIXES,
    IMAGE_SUFFIXES,
    AttachmentRead,
)
from .service import ingest_image, ingest_upload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["Attachments"])


@router.post(
    "/{conversation_id}/attachments",
    response_model=list[AttachmentRead],
    status_code=status.HTTP_201_CREATED,
)
async def upload_attachments(
    conversation_id: uuid.UUID,
    files: list[UploadFile],
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
    embedder=Depends(get_optional_embedder),
):
    """Upload text files (txt/md/csv/pdf/docx/html) into the conversation.

    Files are parsed server-side into conversation-scoped chunks
    (decision 14/21); retrieval reaches them via scoped vector search,
    the `search_files` agent tool, and deterministic context injection.
    """
    conversation = await HistoryRepository(session).get_conversation_for_user(
        user.id, conversation_id
    )
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found."
        )
    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="No files uploaded.",
        )
    image_count = sum(
        1 for f in files
        if Path(f.filename or "").suffix.lower() in IMAGE_SUFFIXES
    )
    if image_count > settings.ATTACHMENT_MAX_IMAGES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"At most {settings.ATTACHMENT_MAX_IMAGES} images per "
                "request."
            ),
        )
    if len(files) > settings.ATTACHMENT_MAX_FILES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"At most {settings.ATTACHMENT_MAX_FILES} files per request."
            ),
        )

    max_bytes = settings.ATTACHMENT_MAX_FILE_MB * 1024 * 1024
    created = []
    for upload in files:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    f"'{upload.filename}': unsupported type. Allowed: "
                    f"txt, md, csv, pdf, docx, html."
                ),
            )
        data = await upload.read()
        if suffix in IMAGE_SUFFIXES:
            created.append(
                await ingest_image(
                    session, conversation_id, user.id,
                    upload.filename or "image.png",
                    upload.content_type or "", data,
                )
            )
            continue
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    f"'{upload.filename}' exceeds "
                    f"{settings.ATTACHMENT_MAX_FILE_MB} MB."
                ),
            )
        created.append(
            await ingest_upload(
                session,
                embedder,
                conversation_id,
                user.id,
                upload.filename or "upload.txt",
                upload.content_type or "",
                data,
            )
        )
    return [
        AttachmentRead(
            id=a.id,
            conversation_id=a.conversation_id,
            filename=a.filename,
            content_type=a.content_type,
            size_bytes=a.size_bytes,
            extracted_chars=a.extracted_chars,
            created_at=a.created_at,
        )
        for a in created
    ]
