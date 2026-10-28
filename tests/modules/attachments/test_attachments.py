"""Offline tests for attachments (PR 15 / decision 14)."""

import json
import uuid

import pytest
from fastapi import HTTPException

from src.modules.attachments.extract import ExtractionError, extract_text
from src.modules.attachments.model import Attachment, AttachmentRef
from src.modules.attachments.service import (
    attachment_context,
    resolve_for_ask,
)

# <----- Extraction ----->


def test_txt_and_csv_extraction():
    assert extract_text("notes.txt", b"hello world") == "hello world"
    csv_text = extract_text("data.csv", b"a,b\n1,2\n3,4")
    assert "a | b" in csv_text and "3 | 4" in csv_text


def test_unsupported_type_is_rejected():
    with pytest.raises(ExtractionError, match="unsupported type"):
        extract_text("payload.exe", b"MZ...")


def test_broken_pdf_degrades_to_a_clear_error():
    with pytest.raises(ExtractionError, match="could not be parsed"):
        extract_text("broken.pdf", b"%PDF-1.4 garbage")


def test_extraction_is_capped(monkeypatch):
    from src.tools.settings import settings

    monkeypatch.setattr(settings, "ATTACHMENT_MAX_CHARS", 10)
    assert extract_text("big.txt", b"x" * 1000) == "x" * 10


# <----- resolve_for_ask ----->


class FakeAttachmentRepo:
    def __init__(self, attachments):
        self.attachments = {a.id: a for a in attachments}

    async def resolve_refs(self, user_id, attachment_ids):
        return {
            a_id: a
            for a_id, a in self.attachments.items()
            if a.user_id == user_id and a_id in set(attachment_ids)
        }


USER_ID = uuid.uuid4()
CONVERSATION_ID = uuid.uuid4()


def _attachment(conversation_id=CONVERSATION_ID) -> Attachment:
    return Attachment(
        id=uuid.uuid4(),
        user_id=USER_ID,
        conversation_id=conversation_id,
        filename="notes.txt",
        size_bytes=10,
        extracted_chars=5,
    )


async def test_requires_conversation_for_scoped_modes():
    with pytest.raises(HTTPException, match="conversation_id"):
        await resolve_for_ask(None, USER_ID, None, [uuid.uuid4()])


async def test_unknown_and_foreign_ids_are_422(monkeypatch):
    att = _attachment()
    other = _attachment(conversation_id=uuid.uuid4())
    from src.modules.attachments import service

    async def resolve_refs(self, user_id, ids):
        return {a.id: a for a in (att, other) if a.id in set(ids)}

    monkeypatch.setattr(
        service.AttachmentRepository, "resolve_refs", resolve_refs
    )

    with pytest.raises(HTTPException, match="Unknown attachment_ids"):
        await resolve_for_ask(
            None, USER_ID, CONVERSATION_ID, [uuid.uuid4()]
        )
    with pytest.raises(HTTPException, match="different conversation"):
        await resolve_for_ask(
            None, USER_ID, CONVERSATION_ID, [other.id], True
        )


async def test_agent_mode_skips_the_conversation_match(monkeypatch):
    att = _attachment(conversation_id=uuid.uuid4())
    from src.modules.attachments import service

    async def resolve_refs(self, user_id, ids):
        return {a.id: a for a in (att,) if a.id in set(ids)}

    monkeypatch.setattr(
        service.AttachmentRepository, "resolve_refs", resolve_refs
    )
    refs = await resolve_for_ask(
        None, USER_ID, None, [att.id], require_same_conversation=False
    )
    assert [r.id for r in refs] == [att.id]


async def test_empty_ids_are_a_noop():
    assert await resolve_for_ask(None, USER_ID, None, []) == []


# <----- Deterministic context injection ----->


class FakeChunksRepo:
    def __init__(self, chunks):
        self.chunks = chunks

    async def chunks_for_attachments(self, ids, limit):
        return self.chunks


async def test_attachment_context_formats_and_caps(monkeypatch):
    from src.tools.settings import settings

    monkeypatch.setattr(settings, "ATTACHMENT_CONTEXT_MAX_CHARS", 50)
    ref = AttachmentRef(
        id=uuid.uuid4(), filename="spec.md", conversation_id=CONVERSATION_ID
    )
    chunks = [
        {"content": "x" * 30, "metadata": {"title": "spec.md"}},
        {"content": "y" * 30, "metadata": {"title": "spec.md"}},
    ]

    class Repo:
        async def chunks_for_attachments(self, ids, limit):
            return chunks

    from src.modules.attachments import service

    def make_repo(session):
        return Repo()

    monkeypatch.setattr(service, "AttachmentRepository", make_repo)
    context = await attachment_context(None, [ref])

    assert context.startswith("[File: spec.md]")
    assert "x" * 30 in context and "y" * 30 not in context  # capped


# <----- Toolbox tool ----->


async def test_search_files_uses_strict_conversation_scope(monkeypatch):
    from src.core.tools.toolbox import AskToolbox, build_tools

    conversation_id = uuid.uuid4()
    captured = {}

    class FakeEmbedder:
        async def embed_texts(self, texts):
            return [[0.1] * 4]

    class FakeChunkRepo:
        async def conversation_chunks_strict(self, cid, embedding, k):
            captured["cid"] = cid
            captured["k"] = k
            return [
                {"content": "passage", "metadata": {"title": "notes.md"}}
            ]

    toolbox = AskToolbox(
        None, None, None, FakeEmbedder(), FakeChunkRepo(),
        attachments=[
            {
                "id": uuid.uuid4(),
                "filename": "notes.md",
                "conversation_id": conversation_id,
            }
        ],
    )
    tool = next(t for t in build_tools(toolbox) if t.name == "search_files")
    result = await tool.coroutine(query="ownership")

    assert captured["cid"] == conversation_id
    assert captured["k"] > 0
    payload = json.loads(result)
    assert payload["files"][0]["source"] == "notes.md"


def test_search_files_absent_without_attachments():
    from src.core.tools.toolbox import AskToolbox, build_tools

    toolbox = AskToolbox(None, None, None, None, None)
    assert all(t.name != "search_files" for t in build_tools(toolbox))
