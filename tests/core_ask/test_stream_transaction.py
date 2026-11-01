"""Streaming routes must not hold a DB transaction while they stream.

Webapp API-NOTES #10: the request session's auth transaction stayed open
for the lifetime of every SSE stream. A dev day of reconnecting job-event
streams parked 15 idle-in-transaction connections on the auth lookup,
saturated the pool, and 500'd every DB-touching route. These integration
tests run the real routes against the live test DB and assert the request
session ends with no open transaction.
"""

import asyncio
import json
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from src.app import app
from src.modules.identity.auth.service import create_access_token
from src.modules.identity.user.model import UserCreate
from src.modules.identity.user.service import create_user
from src.tools.db import get_session
from tests.core_ask.fakes import (
    FakeRanker,
    FakeSearchProvider,
    make_model,
    make_scraper,
)

pytestmark = pytest.mark.integration

# Any stream that outlives this is a hang, not a slow test.
STREAM_BUDGET_S = 30


def _override_session(db_session):
    async def override():
        yield db_session

    app.dependency_overrides[get_session] = override


async def _make_user(db_session):
    suffix = uuid.uuid4().hex[:8]
    user = await create_user(
        db_session,
        UserCreate(
            email=f"{suffix}@example.com",
            nickname=f"user_{suffix}",
            password="Sup3rSecret!pass",
        ),
    )
    return user, create_access_token({"sub": str(user.id)})


def _parse_sse_frames(chunks: list[bytes]) -> list[tuple[str, dict]]:
    events = []
    buffer = b"".join(chunks).decode()
    for block in buffer.split("\n\n"):
        if not block.strip():
            continue
        data = [
            line.split("data: ", 1)[1]
            for line in block.splitlines()
            if line.startswith("data: ")
        ]
        name = [
            line.split("event: ", 1)[1]
            for line in block.splitlines()
            if line.startswith("event: ")
        ]
        if data and name:
            events.append((name[0], json.loads(data[0])))
    return events


async def test_ask_search_stream_ends_with_no_open_transaction(
    db_session, monkeypatch
):
    from src.modules.decision.factory import get_optional_decision_maker
    from src.modules.embedding.factory import get_optional_embedder
    from src.modules.jobs.gates import TurnResources
    from src.modules.llm_provider.dependency import (
        get_llm_client,
        get_turn_resources,
    )
    from src.modules.owl_reranker.factory import get_ranker
    from src.modules.scrapy.factory import get_scraper
    from src.modules.search_engine.factory import get_search_provider
    from src.tools.settings import settings

    monkeypatch.setattr(settings, "AGENTIC_SEARCH", False)
    _user, token = await _make_user(db_session)

    _override_session(db_session)
    app.dependency_overrides[get_search_provider] = lambda: FakeSearchProvider()
    app.dependency_overrides[get_scraper] = make_scraper
    app.dependency_overrides[get_ranker] = lambda: FakeRanker()
    app.dependency_overrides[get_optional_embedder] = lambda: None
    app.dependency_overrides[get_optional_decision_maker] = lambda: None
    app.dependency_overrides[get_llm_client] = lambda: make_model()
    app.dependency_overrides[get_turn_resources] = lambda: TurnResources(keys=[])

    transport = ASGITransport(app=app)
    try:
        async with asyncio.timeout(STREAM_BUDGET_S):
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as ac:
                chunks: list[bytes] = []
                async with ac.stream(
                    "POST",
                    "/ask/search",
                    json={"query": "what is rust?", "stream": True},
                    headers={"Authorization": f"Bearer {token}"},
                ) as response:
                    assert response.status_code == 200
                    async for chunk in response.aiter_bytes():
                        chunks.append(chunk)

        events = _parse_sse_frames(chunks)
        assert events, "the stream produced no events"
        assert events[-1][0] == "done", events[-1]

        assert db_session.in_transaction() is False
    finally:
        app.dependency_overrides.clear()


async def test_job_events_stream_ends_with_no_open_transaction(
    db_session,
):
    user, token = await _make_user(db_session)
    _override_session(db_session)

    transport = ASGITransport(app=app)
    try:
        # The route dispatches as soon as the response headers arrive; the
        # body stays silent until a job event fires, so the stream never
        # terminates on its own. Exercise the real-world scenario — an
        # open-but-idle stream abandoned by the client: after the idle
        # window the cancellation tears the stream down (as a server
        # would on disconnect) and the leak check runs against the
        # request session it used.
        abandoned = False
        try:
            async with asyncio.timeout(5):
                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as ac:
                    async with ac.stream(
                        "GET",
                        "/jobs/events",
                        headers={"Authorization": f"Bearer {token}"},
                    ) as response:
                        assert response.status_code == 200
                        assert user.id  # auth resolved for this stream
                        async for _chunk in response.aiter_bytes():
                            pass  # no event ever arrives
        except TimeoutError:
            abandoned = True

        assert abandoned, "the idle stream should have been abandoned"
        assert db_session.in_transaction() is False
    finally:
        app.dependency_overrides.clear()
