"""End-to-end: a persisted interrupted research run resumes and finishes."""

import uuid

import pytest
import pytest_asyncio

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def env(db_session):
    from httpx import ASGITransport, AsyncClient

    from src.app import app
    from src.modules.identity.auth.service import create_access_token
    from src.modules.identity.user.model import UserCreate
    from src.modules.identity.user.service import create_user
    from src.tools.db import get_session

    row = await create_user(
        db_session,
        UserCreate(
            email=f"{uuid.uuid4().hex[:8]}@e.com",
            nickname=f"u{uuid.uuid4().hex[:6]}",
            password="Sup3rSecret!pass",
        ),
    )
    token = create_access_token({"sub": str(row.id)})
    headers = {"Authorization": f"Bearer {token}"}

    async def override_get_session():
        yield db_session

    app.dependency_overrides[get_session] = override_get_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as ac:
        yield ac, row, headers, app
    app.dependency_overrides.clear()


async def test_resumed_research_run_skips_covered_leg(env, db_session, monkeypatch):
    ac, user, headers, app = env
    from src.modules.jobs.model import Job, JobStatus
    from src.modules.jobs.queue import JobQueue
    from src.modules.jobs.resume import requeue_orphans
    from src.modules.notes.model import ResearchNote
    from tests.conftest import _integration_database_url

    monkeypatch.setattr(
        __import__("src.modules.jobs.resume", fromlist=["settings"]).settings,
        "JOB_MAX_RETRIES",
        3,
    )
    import asyncio

    from src.tools.settings import settings

    covered_task = "already-gathered leg"
    note = ResearchNote(
        id=uuid.uuid4(),
        user_id=user.id,
        sequence=0,
        task=covered_task,
        content="body",
        sources_used=["https://example.com/seed"],
    )
    db_session.add(note)
    await db_session.commit()

    job = Job(
        user_id=user.id,
        kind="research",
        payload={
            "query": "resumed query",
            "research": {"note_ids": [str(note.id)]},
        },
    )
    job.status = JobStatus.RUNNING
    db_session.add(job)
    await db_session.commit()
    await db_session.refresh(job)

    # Stub the model/tool resolution the rebuilt runner performs.
    import src.modules.jobs.resume as resume_mod
    from src.modules.search_engine.providers.base import ISearchProvider
    from tests.core_ask.fakes import (
        FakeRanker,
        ScriptedToolModel,
        make_scraper,
        tool_call,
    )
    scripted = ScriptedToolModel(
        responses=[
            tool_call(
                "ResearchPlan",
                {"subquestions": [covered_task, "fresh leg"]},
            ),
            tool_call(
                "ResearchReport",
                {"report": "## Covered\nSynthesis from both legs."},
            ),
            # Spare: the editorial step consumes one response on resume,
            # the fallback report generator another.
            tool_call(
                "ResearchReport",
                {"report": "## Covered\nSynthesis from both legs."},
            ),
        ]
    )

    class _StubResolved:
        provider_id = None
        max_concurrency = 1
        capabilities = ["tools", "thinking"]
        model_name = "scripted"
        provider_type = "ollama"

    async def fake_chat(session, job):
        return scripted, _StubResolved()

    class _FakeResolved:
        pass

    async def _fake_from_resolved(resolved):
        return FakeRanker()

    async def fake_resolve_ranker(session, model_id=None):
        return _FakeResolved()

    class FakeSearch(ISearchProvider):
        async def search(self, query, engines=None, categories=None):
            from src.modules.search_engine.providers.base import SearchResult

            return [
                SearchResult("t", "rust snippet", "https://example.com/fresh")
            ]

    monkeypatch.setattr(resume_mod, "_chat_model", fake_chat)
    monkeypatch.setattr(settings, "RESULT_SANITY_FILTER", False)
    async def _fake_scrap(session):
        return make_scraper()

    monkeypatch.setattr(
        "src.modules.scrapy.factory.ScraperFactory.create",
        staticmethod(_fake_scrap),
    )
    monkeypatch.setattr(
        "src.modules.search_engine.factory.SearchProviderFactory.create_provider",
        staticmethod(lambda *a, **k: _fut_search()),
    )


    class _FakeProvider(ISearchProvider):
        async def search(self, query, engines=None, categories=None):
            return _fake_results()

    def _fake_results():
        from src.modules.search_engine.providers.base import SearchResult

        return [SearchResult("r", "rust snippet", "https://example.com/fresh")]

    async def _fut_search():
        return _FakeProvider()

    monkeypatch.setattr(
        "src.modules.search_engine.factory.SearchProviderFactory.create_provider",
        staticmethod(lambda *a, **k: _fut_search()),
    )

    monkeypatch.setattr(
        "src.modules.owl_reranker.factory._resolve_ranker_model",
        fake_resolve_ranker,
    )
    monkeypatch.setattr(
        "src.modules.owl_reranker.factory.RankerFactory.from_resolved",
        staticmethod(_fake_from_resolved),
    )

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    test_engine = create_async_engine(_integration_database_url())
    maker = async_sessionmaker(bind=test_engine, expire_on_commit=False)

    queue = JobQueue(session_factory=lambda: db_session)

    async def registry(job_row):
        from src.modules.jobs.resume import build_runner

        return await build_runner(job_row, db_session)

    async def jobs_provider():
        return [job]

    requeued = await requeue_orphans(
        queue, session_factory=lambda: maker(), jobs_provider=jobs_provider,
        registry=registry,
    )
    assert requeued == 1
    assert job.status == JobStatus.QUEUED
    queue.start()
    try:
        for _ in range(200):
            if job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
                break
            await asyncio.sleep(0.05)
        assert job.status == JobStatus.SUCCEEDED, (job.error, job.payload)
        # Synthesis covers the seeded (pre-restart) note: the report is
        # the scripted one, not the no-content fallback.
        assert "both legs" in (job.result_summary or ""), job.result_summary
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
        from sqlmodel import select as _sel

        count = len((await db_session.exec(_sel(ResearchNote))).all())
        assert count == 1  # no duplicate note for the covered leg
    finally:
        await queue.stop()
        await test_engine.dispose()
