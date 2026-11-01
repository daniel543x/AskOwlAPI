"""Regression: succeeded background jobs must carry `conversation_id` (API-NOTES #10).

The webapp InBox links a done job to the full answer conversation; rows
persisted `null` break that. These tests run the real enqueue runners
through a persisted JobQueue against the live test DB and assert the
succeeded rows link a conversation holding the saved turn.
"""

import asyncio
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import select

from src.modules.embedding.base import IEmbedder
from src.modules.history.model import Conversation, Message
from src.modules.identity.user.model import UserCreate
from src.modules.identity.user.service import create_user
from src.modules.jobs.model import Job, JobStatus
from src.modules.jobs.queue import JobQueue
from tests.core_ask.fakes import (
    FakeRanker,
    FakeSearchProvider,
    ScriptedToolModel,
    make_scraper,
    tool_call,
)

pytestmark = pytest.mark.integration

QUERY = "memory safety in rust"
REPORT = "## Findings\nRust prevents data races at compile time."


class FixedEmbedder(IEmbedder):
    """Deterministic embedder so chunk rows are actually written."""

    @property
    def dims(self) -> int:
        return 4

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def _search_result():
    from src.modules.search_engine.providers.base import SearchResult

    return SearchResult(
        "Result A",
        "Rust prevents data races at compile time.",
        "http://example.com/a",
    )


async def _wait_terminal(session_factory, job_id, ticks: int = 600):
    row = None
    for _ in range(ticks):
        async with session_factory() as session:
            row = await session.get(Job, job_id)
            if row is not None and row.status in (
                JobStatus.SUCCEEDED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            ):
                return row
        await asyncio.sleep(0.05)
    return row


async def _assert_linked_conversation(session_factory, job):
    """The succeeded row links a conversation that holds the saved turn."""
    assert job.status == JobStatus.SUCCEEDED, job.error
    async with session_factory() as session:
        row = await session.get(Job, job.id)
        assert row is not None and row.conversation_id is not None
        conversation = await session.get(Conversation, row.conversation_id)
        assert conversation is not None
        messages = list(
            (
                await session.execute(
                    select(Message).where(
                        Message.conversation_id == conversation.id
                    )
                )
            ).scalars()
        )
        assert {m.role for m in messages} == {"user", "assistant"}


async def _make_env(db_session):
    suffix = uuid.uuid4().hex[:8]
    user = await create_user(
        db_session,
        UserCreate(
            email=f"{suffix}@example.com",
            nickname=f"user_{suffix}",
            password="Sup3rSecret!pass",
        ),
    )
    from tests.conftest import _integration_database_url

    test_engine = create_async_engine(_integration_database_url())
    session_factory = async_sessionmaker(
        bind=test_engine, class_=AsyncSession, expire_on_commit=False
    )
    return user, test_engine, session_factory


async def test_background_search_job_links_conversation(db_session, monkeypatch):
    from src.core.ask.router import SearchRequest, _enqueue_search_turn
    from src.modules.jobs.gates import TurnResources
    from src.tools.settings import settings

    monkeypatch.setattr(settings, "AGENTIC_SEARCH", False)

    user, test_engine, session_factory = await _make_env(db_session)
    queue = JobQueue(session_factory=session_factory)
    queue.start()
    try:
        scripted = ScriptedToolModel(
            responses=[tool_call("CitedAnswer", {"answer": "Same."})]
        )
        response = await _enqueue_search_turn(
            SearchRequest(query=QUERY, rewrite="none"),
            scripted,
            FakeSearchProvider(results=[_search_result()]),
            make_scraper(),
            FakeRanker(),
            FixedEmbedder(),
            None,
            user,
            TurnResources(keys=[]),
            queue=queue,
            run_engine=test_engine,
        )
        assert response.status_code == 202
        import json

        job_id = uuid.UUID(json.loads(bytes(response.body))["job_id"])
        job = await _wait_terminal(session_factory, job_id)
        assert job is not None
        assert job.status == JobStatus.SUCCEEDED, job.error
    finally:
        await queue.stop()
        await test_engine.dispose()

    await _assert_linked_conversation(session_factory, job)


async def test_background_research_job_links_conversation(
    db_session, monkeypatch
):
    from src.core.research.router import _enqueue_research_turn
    from src.modules.jobs.gates import TurnResources
    from src.tools.settings import settings

    monkeypatch.setattr(settings, "RESULT_SANITY_FILTER", False)

    user, test_engine, session_factory = await _make_env(db_session)
    queue = JobQueue(session_factory=session_factory)
    queue.start()
    try:
        scripted = ScriptedToolModel(
            responses=[
                tool_call("ResearchPlan", {"subquestions": ["rust guarantees"]}),
                tool_call("ResearchReport", {"report": REPORT}),
                tool_call("ResearchReport", {"report": REPORT}),
                tool_call("ResearchReport", {"report": REPORT}),
                tool_call("ResearchReport", {"report": REPORT}),
            ]
        )
        response = await _enqueue_research_turn(
            QUERY,
            FakeSearchProvider(results=[_search_result()]),
            scripted,
            make_scraper(),
            FakeRanker(),
            None,
            db_session,
            user,
            [],
            TurnResources(keys=[]),
            [],
            None,
            queue=queue,
            run_engine=test_engine,
        )
        assert response.status_code == 202
        import json

        job_id = uuid.UUID(json.loads(bytes(response.body))["job_id"])
        job = await _wait_terminal(session_factory, job_id)
        assert job is not None
        assert job.status == JobStatus.SUCCEEDED, job.error
    finally:
        await queue.stop()
        await test_engine.dispose()

    await _assert_linked_conversation(session_factory, job)
