"""Regression: re-running the same background search must succeed (API-NOTES #9).

The webapp reported UniqueViolationError jobs on re-runs whose chunks/turn
already existed. This test is the reproduction vehicle: it runs the real
pipeline twice through a persisted JobQueue against the live test DB.
"""

import asyncio
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.ask.services.search import search_json_adapter
from src.modules.embedding.base import IEmbedder
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


def _make_runner(session_factory, user_id):
    async def runner() -> str:
        async with session_factory() as session:
            result = await search_json_adapter(
                QUERY,
                FakeSearchProvider(results=[_search_result()]),
                ScriptedToolModel(
                    responses=[tool_call("CitedAnswer", {"answer": "Same answer."})]
                ),
                make_scraper(),
                FakeRanker(),
                embedder=FixedEmbedder(),
                session=session,
                user_id=user_id,
                rewrite="none",
            )
        return (result.get("answer") or "")[:4000]

    return runner


async def _wait_terminal(job: Job, ticks: int = 600) -> None:
    for _ in range(ticks):
        if job.status in (
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        ):
            return
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_background_search_rerun_succeeds(db_session):
    from tests.conftest import _integration_database_url

    suffix = uuid.uuid4().hex[:8]
    user = await create_user(
        db_session,
        UserCreate(
            email=f"{suffix}@example.com",
            nickname=f"user_{suffix}",
            password="Sup3rSecret!pass",
        ),
    )

    test_engine = create_async_engine(_integration_database_url())
    session_factory = async_sessionmaker(
        bind=test_engine, class_=AsyncSession, expire_on_commit=False
    )
    queue = JobQueue(session_factory=session_factory)
    queue.start()
    try:
        statuses = []
        for _ in range(2):
            job = Job(user_id=user.id, kind="search", payload={"query": QUERY})
            await queue.submit(job, _make_runner(session_factory, user.id))
            await _wait_terminal(job)
            statuses.append(job.status)
            assert job.status == JobStatus.SUCCEEDED, job.error
    finally:
        await queue.stop()
        await test_engine.dispose()

    assert statuses == [JobStatus.SUCCEEDED, JobStatus.SUCCEEDED]

    async with session_factory() as session:
        chunk_count = (
            (await session.execute(text("SELECT count(*) FROM chunk"))).scalar_one()
        )
    assert chunk_count > 0
