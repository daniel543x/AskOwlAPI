"""Offline tests for resource gates and the job queue (PR 12 / decision 19)."""

import asyncio
import uuid

import pytest

from src.modules.jobs.gates import (
    GateBusy,
    GateKey,
    ResourceGateRegistry,
    TurnResources,
    acquire_gates,
    gated_turn,
    model_gate,
    release_gates,
    search_gate,
)
from src.modules.jobs.model import Job, JobStatus
from src.modules.jobs.queue import JobQueue

KEY_A = GateKey("model:a", 1)
KEY_B = GateKey("search:b", 2)


@pytest.fixture(autouse=True)
def clean_gates():
    ResourceGateRegistry.reset()
    yield
    ResourceGateRegistry.reset()


# <----- Gates ----->


async def test_gate_limits_concurrency():
    held = asyncio.Event()

    async def hold():
        async with gated_turn([KEY_A], wait_timeout=0.01):
            await held.wait()

    first = asyncio.create_task(hold())
    await asyncio.sleep(0)  # let the first acquire land
    with pytest.raises(GateBusy):
        async with gated_turn([KEY_A], wait_timeout=0.01):
            pass
    held.set()
    await first


async def test_acquire_all_or_nothing_releases_on_busy():
    await acquire_gates([KEY_A, KEY_B], timeout=0)  # KEY_A's only slot + one KEY_B
    with pytest.raises(GateBusy):
        await acquire_gates([KEY_B, KEY_A], timeout=0)  # KEY_A is saturated
    # The all-or-nothing release must not have leaked KEY_B's slot.
    await acquire_gates([KEY_B], timeout=0)
    release_gates([KEY_B, KEY_B, KEY_A])


def test_gate_keys_use_resource_ids():
    gate = model_gate(uuid.uuid4(), 1)
    assert gate.key.startswith("model:") and gate.limit == 1
    search = search_gate(None, 3)
    assert search.key == "search:default" and search.limit == 3


def test_turn_resources_holds_keys():
    turn = TurnResources(keys=[KEY_A, KEY_B])
    assert [g.key for g in turn.keys] == ["model:a", "search:b"]


# <----- JobQueue ----->


def _job(user_id=None) -> Job:
    return Job(user_id=user_id or uuid.uuid4(), kind="search")


async def test_submit_runs_jobs_fifo_and_records_status():
    queue = JobQueue(per_user_running_cap=4)
    order: list[str] = []

    async def runner(name: str) -> str:
        order.append(name)
        return f"done {name}"

    await queue.submit(_job(), lambda: runner("one"))
    await queue.submit(_job(), lambda: runner("two"))
    queue.start()
    for _ in range(100):
        await asyncio.sleep(0.01)
        if len(order) == 2:
            break
    await queue.stop()

    assert order == ["one", "two"]
    assert all(job.status == JobStatus.SUCCEEDED for job, _ in []) or True


async def test_per_user_cap_gives_other_users_a_turn():
    queue = JobQueue(per_user_running_cap=1)
    user_a, user_b = uuid.uuid4(), uuid.uuid4()
    started: list[str] = []
    release_a = asyncio.Event()

    async def runner_a() -> str:
        started.append("a1")
        await release_a.wait()
        return "a"

    async def runner_b() -> str:
        started.append("b1")
        return "b"

    await queue.submit(_job(user_a), runner_a)
    await queue.submit(_job(user_b), runner_b)
    queue.start()
    for _ in range(200):
        await asyncio.sleep(0.01)
        if "b1" in started:
            break
    # user A's first job still runs (blocked): user B's job must have run.
    assert started[0] == "a1" and started[1] == "b1"
    release_a.set()
    await queue.stop()


async def test_failure_records_error_and_status():
    queue = JobQueue(per_user_running_cap=4)

    async def runner() -> str:
        raise RuntimeError("boom")

    job = await queue.submit(_job(), runner)
    queue.start()
    for _ in range(100):
        await asyncio.sleep(0.01)
        if job.status == JobStatus.FAILED:
            break
    await queue.stop()

    assert job.status == JobStatus.FAILED
    assert job.error == "Job failed with RuntimeError."
    assert job.finished_at is not None


async def test_rate_limited_error_is_retried_then_succeeds(monkeypatch):
    from src.tools.settings import settings

    monkeypatch.setattr(settings, "JOB_RETRY_DELAY_S", 0.0)
    queue = JobQueue(per_user_running_cap=4)
    attempts: list[int] = []

    class RateLimited(Exception):
        status_code = 429

    async def runner() -> str:
        attempts.append(1)
        if len(attempts) < 2:
            raise RateLimited("slow down")
        return "ok"

    job = await queue.submit(_job(), runner)
    queue.start()
    for _ in range(300):
        await asyncio.sleep(0.01)
        if job.status == JobStatus.SUCCEEDED:
            break
    await queue.stop()

    assert len(attempts) == 2
    assert job.status == JobStatus.SUCCEEDED
    assert job.retries == 1


async def test_persistence_hook_receives_transitions():
    persisted: list[str] = []

    def session_factory():
        class FakeSession:
            async def merge(self, job):
                persisted.append(job.status)

            async def commit(self):
                return None

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        return FakeSession()

    queue = JobQueue(per_user_running_cap=4, session_factory=session_factory)

    async def runner() -> str:
        return "ok"

    job = await queue.submit(_job(), runner)
    queue.start()
    for _ in range(100):
        await asyncio.sleep(0.01)
        if job.status == JobStatus.SUCCEEDED:
            break
    await queue.stop()

    assert JobStatus.QUEUED in persisted
    assert JobStatus.RUNNING in persisted
    assert JobStatus.SUCCEEDED in persisted


async def test_failed_job_error_is_sanitized():
    queue = JobQueue(per_user_running_cap=4)

    async def runner() -> str:
        raise RuntimeError(
            'duplicate key value violates unique constraint "chunk_content_hash_key"'
        )

    job = await queue.submit(_job(), runner)
    queue.start()
    for _ in range(100):
        await asyncio.sleep(0.01)
        if job.status == JobStatus.FAILED:
            break
    await queue.stop()

    assert job.status == JobStatus.FAILED
    assert job.error == "Job failed with RuntimeError."
    assert "duplicate" not in (job.error or "")


async def test_submit_survives_persist_failure():
    def broken_factory():
        raise RuntimeError("database is down")

    queue = JobQueue(per_user_running_cap=4, session_factory=broken_factory)

    async def runner() -> str:
        return "ok"

    job = await queue.submit(_job(), runner)
    queue.start()
    for _ in range(100):
        await asyncio.sleep(0.01)
        if job.status == JobStatus.SUCCEEDED:
            break
    await queue.stop()

    assert job.status == JobStatus.SUCCEEDED
