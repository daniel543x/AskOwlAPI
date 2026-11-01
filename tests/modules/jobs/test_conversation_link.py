"""Offline tests for stamping a job row with its turn's conversation.

Background jobs must carry `conversation_id` on the succeeded row so the
InBox can link to the full answer conversation (webapp API-NOTES #10).
"""

import uuid

from src.modules.jobs.model import Job, link_conversation


def test_result_conversation_id_is_stamped():
    job = Job(user_id=uuid.uuid4(), kind="search")
    cid = uuid.uuid4()
    link_conversation(job, {"conversation_id": str(cid)})
    assert job.conversation_id == cid


def test_result_wins_over_the_request_fallback():
    job = Job(user_id=uuid.uuid4(), kind="search")
    created = uuid.uuid4()
    requested = uuid.uuid4()
    link_conversation(job, {"conversation_id": str(created)}, requested)
    assert job.conversation_id == created


def test_missing_result_falls_back_to_the_request_conversation():
    job = Job(user_id=uuid.uuid4(), kind="search")
    requested = uuid.uuid4()
    link_conversation(job, {"conversation_id": None}, requested)
    assert job.conversation_id == requested


def test_unparsable_result_falls_back():
    job = Job(user_id=uuid.uuid4(), kind="search")
    requested = uuid.uuid4()
    link_conversation(job, {"conversation_id": "not-a-uuid"}, requested)
    assert job.conversation_id == requested


def test_nothing_known_leaves_the_row_null():
    job = Job(user_id=uuid.uuid4(), kind="research")
    link_conversation(job, {})
    assert job.conversation_id is None
