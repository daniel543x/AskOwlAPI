"""Offline + integration tests for rebuilding and re-enqueueing runners."""

import uuid
from types import SimpleNamespace

import pytest

from src.modules.identity import role as _role  # noqa: F401
from src.modules.jobs.model import Job, JobStatus
from src.modules.jobs.queue import JobQueue
from src.modules.jobs.resume import requeue_orphans


class RecordingRunner:
    def __init__(self):
        self.calls: list[Job] = []

    def runner_for(self, job):
        async def runner() -> str:
            self.calls.append(job)
            return "done"
        return runner


@pytest.mark.asyncio
async def test_search_job_replays_whole_turn():
    """kind=search rebuilds a runner that replays the saved turn."""
    session = SimpleNamespace(bind=None)
    from src.modules.jobs.resume import build_runner

    job = Job(
        user_id=uuid.uuid4(), kind="search",
        payload={"query": "rust", "conversation_id": None},
    )
    runner = await build_runner(job, session)
    assert callable(runner)


@pytest.mark.asyncio
async def test_unknown_kind_builds_none():
    from src.modules.jobs.resume import build_runner

    session = SimpleNamespace(bind=None)
    job = Job(user_id=uuid.uuid4(), kind="ggdeals", payload={})
    assert await build_runner(job, session) is None


@pytest.mark.asyncio
async def test_requeue_orphans_runs_and_respects_retries(caplog):

    reg = RecordingRunner()

    async def registry(job):
        # Both kinds are resumable with a real DB set of deps; the
        # offline stub simply hands back a runner for each.
        return reg.runner_for(job)

    jobs = [
        Job(user_id=uuid.uuid4(), kind="search", payload={"query": "x"}),
        Job(user_id=uuid.uuid4(), kind="research", payload={"query": "x"}),
        Job(user_id=uuid.uuid4(), kind="search", payload={"query": "y"}),
    ]
    jobs[2].retries = 1_000_000  # way past the cap
    statuses = [JobStatus.RUNNING] * 3
    for job, status in zip(jobs, statuses, strict=True):
        job.status = status

    queue = JobQueue(per_user_running_cap=4)
    persisted: list = []

    class StubSession:
        async def merge(self, job):
            persisted.append(job)
            return job

        async def commit(self):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    def stub_factory():
        return StubSession()

    async def provider():
        return jobs

    sought = await requeue_orphans(
        queue,
        session_factory=stub_factory,
        jobs_provider=provider,
        registry=registry,
    )
    assert persisted  # the capped job's failed state was persisted
    assert sought == 2  # capped job still fails, the others requeue
    assert tuple(j.status for j in jobs) == (
        JobStatus.QUEUED,
        JobStatus.QUEUED,
        JobStatus.FAILED,
    )


def statuses_count(jobs):
    return tuple(j.status for j in jobs)
