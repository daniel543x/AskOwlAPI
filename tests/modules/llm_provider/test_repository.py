"""Regression tests for ModelRepository against a live Postgres (integration)."""

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession

from src.modules.llm_provider import repository as repo_module
from src.modules.llm_provider.model import ModelProvider
from src.modules.llm_provider.repository import ModelRepository

pytestmark = pytest.mark.integration


async def _expiring_session(db_session: AsyncSession) -> AsyncSession:
    """A session configured like production: instances expire on commit."""
    factory = async_sessionmaker(
        bind=db_session.bind, class_=AsyncSession, expire_on_commit=True
    )
    return factory()


async def test_probe_providers_returns_unexpired_instances(
    db_session, monkeypatch
):
    """GET /health/models builds response models from the returned providers.

    Before the fix, probe_providers' commit expired every instance and the
    route's first attribute access raised MissingGreenlet (HTTP 500).
    """
    async def fake_probe(provider_type: str, base_url, timeout: float) -> str:
        return "up"

    monkeypatch.setattr(repo_module, "probe_provider", fake_probe)

    session = await _expiring_session(db_session)
    try:
        session.add(
            ModelProvider(
                name="probe-regression",
                provider_type="ollama",
                base_url="http://127.0.0.1:1",
                is_active=True,
            )
        )
        await session.commit()

        repo = ModelRepository(session)
        probed = await repo.probe_providers()

        matches = [p for p in probed if p.name == "probe-regression"]
        assert matches, "seeded provider missing from probe results"
        for provider in probed:
            # Attribute access on expired instances would raise MissingGreenlet.
            assert provider.id is not None
            assert provider.health_status in {"up", "down", "unknown", "local"}
        assert matches[0].health_status == "up"
    finally:
        await session.close()


async def test_probe_providers_skips_inactive(db_session, monkeypatch):
    async def fail_probe(provider_type: str, base_url, timeout: float) -> str:
        raise AssertionError("inactive providers must not be probed")

    monkeypatch.setattr(repo_module, "probe_provider", fail_probe)

    session = await _expiring_session(db_session)
    try:
        session.add(
            ModelProvider(
                name="probe-inactive",
                provider_type="ollama",
                base_url=None,
                is_active=False,
            )
        )
        await session.commit()

        repo = ModelRepository(session)
        probed = await repo.probe_providers()

        matches = [p for p in probed if p.name == "probe-inactive"]
        assert matches, "inactive provider still listed"
        assert matches[0].health_status != "up"
    finally:
        await session.close()
