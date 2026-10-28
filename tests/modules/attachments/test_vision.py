"""Offline tests for image attachments + vision gating (PR 16 / decision 14)."""

import pytest
from fastapi import HTTPException

from src.modules.attachments.model import (
    ATTACHMENT_KIND_IMAGE,
    ATTACHMENT_KIND_TEXT,
    AttachmentRef,
)
from src.modules.attachments.service import ensure_vision


def _ref(kind: str = ATTACHMENT_KIND_TEXT) -> AttachmentRef:
    import uuid

    return AttachmentRef(
        id=uuid.uuid4(),
        filename="photo.png" if kind == ATTACHMENT_KIND_IMAGE else "notes.txt",
        conversation_id=uuid.uuid4(),
        kind=kind,
    )


# <----- Vision gate ----->


def test_image_without_vision_is_a_coded_422():
    with pytest.raises(HTTPException) as exc_info:
        ensure_vision(["tools"], [_ref(ATTACHMENT_KIND_IMAGE)])
    assert exc_info.value.status_code == 422
    detail = exc_info.value.detail
    assert detail["code"] == "model_lacks_vision"
    assert "photo.png" in detail["message"]


def test_text_attachments_need_no_vision():
    ensure_vision(["tools"], [_ref(ATTACHMENT_KIND_TEXT)])


def test_vision_model_passes_image_turns():
    ensure_vision(["tools", "vision"], [_ref(ATTACHMENT_KIND_IMAGE)])


# <----- Multimodal synthesis blocks ----->


def test_answer_messages_build_image_blocks():
    from langchain_core.messages import HumanMessage

    from src.core.ask.services.llm_steps import _answer_messages

    messages = _answer_messages("q", "ctx", None, None, images=[b"PNG"])
    last = messages[-1]
    assert isinstance(last, HumanMessage)
    assert last.content[0] == {"type": "text", "text": "q"}
    assert last.content[1]["type"] == "image_url"
    assert last.content[1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_answer_messages_without_images_stay_text():

    from src.core.ask.services.llm_steps import _answer_messages

    messages = _answer_messages("q", "ctx")
    assert messages[-1].content == "q"


def test_report_messages_build_image_blocks():
    from src.core.research.services.synthesis import _report_messages

    messages = _report_messages("q", [], None, images=[b"PNG"])
    assert messages[-1].content[1]["type"] == "image_url"


def test_image_blocks_base64_roundtrip():
    from src.core.research.services.agent import image_blocks_for

    blocks = image_blocks_for([b"abcd"])
    assert blocks[0]["image_url"]["url"].endswith("YWJjZA==")


# <----- Blob sweep ----->


async def test_blob_sweep_counts_orphans():
    from src.modules.attachments.repository import AttachmentRepository

    class FakeResult:
        rowcount = 3

    class FakeSession:
        async def execute(self, statement, params=None):
            return FakeResult()

        async def commit(self):
            return None

    swept = await AttachmentRepository(FakeSession()).sweep_orphan_image_blobs()
    assert swept == 3
