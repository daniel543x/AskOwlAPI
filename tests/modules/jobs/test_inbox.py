"""Offline tests for the Inbox layer (PR 13): bus, cancel, routes."""

import asyncio
import uuid
from dataclasses import dataclass, field

import pytest
from httpx import ASGITransport, AsyncClient

from src.app import app
from src.modules.identity.auth.security import get_current_user
from src.modules.jobs.events import JobEventBus
from src.modules.jobs.model import Job, JobStatus
from src.modules.jobs.queue import JobQueue
from src.modules.jobs.router import get_repository

USER_ID = uuid.uuid4()


def _as_user():
    from typing import cast

    from src.modules.identity.user.model import User

    return cast(User, cast(object, FakeUser(id=USER_ID, roles=[])))


@dataclass
class FakeUser:
    id: uuid.UUID
    roles: list = field(default_factory=list)


@dataclass
class FakeJobRepo:
    jobs: dict = field(default_factory=dict)

    async def list_for_user(self, user_id, seen=None):
        out = [j for j in self.jobs.values() if j.user_id == user_id]
        if seen is not None:
            out = [j for j in out if j.seen == seen]
        return sorted(out, key=lambda j: j.created_at or 0, reverse=True)

    async def get_for_user(self, job_id, user_id):
        job = self.jobs.get(job_id)
        if job is None or job.user_id != user_id:
            return None
        return job

    async def mark_seen(self, job):
        job.seen = True
        return job

    async def mark_cancelled(self, job):
        job.status = JobStatus.CANCELLED
        return job

    async def sweep_orphans(self):
        orphans = [
            j for j in self.jobs.values() if j.status == JobStatus.RUNNING
        ]
        for job in orphans:
            job.status = JobStatus.FAILED
        return len(orphans)


@pytest.fixture
def fake_repo():
    return FakeJobRepo()


@pytest.fixture
async def client(fake_repo):
    app.dependency_overrides[get_repository] = lambda: fake_repo
    app.dependency_overrides[get_current_user] = _as_user
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


def _make_job_row() -> Job:
    return Job(user_id=USER_ID, kind="search")


# <----- Event bus ----->


def test_bus_fans_out_per_user_and_unsubscribes():
    bus = JobEventBus()
    a1 = bus.subscribe(USER_ID)
    a2 = bus.subscribe(USER_ID)
    b1 = bus.subscribe(uuid.uuid4())
    bus.publish(USER_ID, {"status": "running"})

    assert a1.qsize() == 1 and a2.qsize() == 1 and b1.qsize() == 0

    bus.unsubscribe(USER_ID, a1)
    bus.unsubscribe(USER_ID, a2)
    bus.publish(USER_ID, {"status": "done"})
    assert a1.qsize() == 1  # no new events after unsubscribe
    assert bus.listener_count(USER_ID) == 0


# <----- Queue cancel ----->


async def test_cancel_removes_a_queued_job():
    queue = JobQueue(per_user_running_cap=4)

    async def runner() -> str:
        return "never"

    job = await queue.submit(_make_job_row(), runner)
    cancelled = await queue.cancel(job.id)

    assert cancelled is not None
    assert cancelled.status == JobStatus.CANCELLED
    assert queue.queued_jobs() == []
    assert await queue.cancel(uuid.uuid4()) is None


async def test_cancel_reports_running_jobs_as_uncancellable():
    queue = JobQueue(per_user_running_cap=1)
    started = asyncio.Event()

    async def runner() -> str:
        started.set()
        await asyncio.sleep(3600)

    job = await queue.submit(_make_job_row(), runner)
    queue.start()
    await asyncio.wait_for(started.wait(), timeout=2)
    assert await queue.cancel(job.id) is None  # running: not cancellable here
    await queue.stop()


# <----- Routes ----->


async def _make_job(fake_repo: FakeJobRepo, **overrides) -> Job:
    import datetime

    job = Job(
        user_id=USER_ID,
        kind="research",
        status=overrides.pop("status", JobStatus.SUCCEEDED),
        seen=overrides.pop("seen", False),
        result_summary=overrides.pop("result_summary", "## Findings"),
        created_at=overrides.pop("created_at", datetime.datetime.now(datetime.UTC)),
        **overrides,
    )
    fake_repo.jobs[job.id] = job
    return job


async def test_list_jobs_filters_by_seen(client, fake_repo):
    await _make_job(fake_repo)
    await _make_job(fake_repo, seen=True)

    everything = await client.get("/jobs")
    assert everything.status_code == 200
    assert len(everything.json()) == 2

    unread = await client.get("/jobs", params={"seen": False})
    assert [j["seen"] for j in unread.json()] == [False]


async def test_seen_flags_a_job(client, fake_repo):
    job = await _make_job(fake_repo)
    response = await client.post(f"/jobs/{job.id}/seen")
    assert response.status_code == 200
    assert response.json()["seen"] is True


async def test_cancel_queued_job_and_conflicts_on_done(client, fake_repo):
    queued = await _make_job(fake_repo, status=JobStatus.QUEUED)
    done = await _make_job(fake_repo, status=JobStatus.SUCCEEDED)

    response = await client.post(f"/jobs/{queued.id}/cancel")
    assert response.status_code == 200
    assert response.json()["status"] == JobStatus.CANCELLED

    conflict = await client.post(f"/jobs/{done.id}/cancel")
    assert conflict.status_code == 409


async def test_foreign_jobs_are_invisible(client, fake_repo):
    foreign = await _make_job(fake_repo)
    foreign.user_id = uuid.uuid4()

    listing = await client.get("/jobs")
    assert listing.json() == []

    seen = await client.post(f"/jobs/{foreign.id}/seen")
    assert seen.status_code == 404

    cancel = await client.post(f"/jobs/{foreign.id}/cancel")
    assert cancel.status_code == 404


# <----- Startup sweep ----->


async def test_sweep_marks_running_jobs_failed(fake_repo):
    await _make_job(fake_repo, status=JobStatus.RUNNING)
    await _make_job(fake_repo, status=JobStatus.SUCCEEDED)

    swept = await fake_repo.sweep_orphans()

    assert swept == 1
    statuses = [j.status for j in fake_repo.jobs.values()]
    assert JobStatus.FAILED in statuses and JobStatus.SUCCEEDED in statuses


# <----- Live updates ----->


async def test_queue_publishes_transitions_to_the_bus():
    from src.modules.jobs.factory import get_event_bus

    bus = get_event_bus()
    listener = bus.subscribe(USER_ID)
    try:
        queue = JobQueue(per_user_running_cap=4, event_bus=bus)

        async def runner() -> str:
            return "ok"

        await queue.submit(_make_job_row(), runner)
        first = await asyncio.wait_for(listener.get(), timeout=2)
        assert first["status"] == JobStatus.QUEUED

        queue.start()
        update = await asyncio.wait_for(listener.get(), timeout=2)
        while update["status"] != JobStatus.SUCCEEDED:
            update = await asyncio.wait_for(listener.get(), timeout=2)
        assert update["result_summary"] == "ok"
    finally:
        bus.unsubscribe(USER_ID, listener)
