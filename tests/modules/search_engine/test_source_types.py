"""Offline tests for provider-declared source types (PR 11 / decision 13)."""

import uuid

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from src.app import app
from src.modules.identity.auth.security import get_current_user
from src.modules.search_engine.factory import (
    declared_source_types,
    get_declared_source_types,
)
from src.modules.search_engine.source_types import (
    KNOWN_SOURCE_TYPES,
    categories_for,
    ensure_source_types,
    extra_engines_for,
)
from tests.modules.llm_provider.test_router import FakeUser

# <----- Registry mapping ----->


def test_categories_map_web_news_academic():
    assert categories_for(["web"]) == "general"
    assert categories_for(["news", "academic"]) == "news,science"
    assert categories_for(None) is None
    assert categories_for([]) is None


def test_social_maps_to_engines_not_categories():
    assert categories_for(["social"]) is None
    assert extra_engines_for(["social"]) == ["reddit"]


def test_extra_engines_dedupe_and_preserve_order():
    assert extra_engines_for(["social", "social"]) == ["reddit"]
    assert extra_engines_for(["web"]) == []


# <----- Validation ----->


def test_empty_request_always_allowed():
    ensure_source_types([], ["web"])
    ensure_source_types(None, [])


def test_unknown_source_type_is_422():
    with pytest.raises(HTTPException) as exc_info:
        ensure_source_types(["web", "darkweb"], ["web", "news"])
    assert exc_info.value.status_code == 422
    assert "darkweb" in exc_info.value.detail


def test_declared_types_pass():
    ensure_source_types(list(KNOWN_SOURCE_TYPES), list(KNOWN_SOURCE_TYPES))


# <----- declared_source_types helper ----->


class _NoConfigResult:
    def scalar_one_or_none(self):
        return None


class _NoConfigSession:
    async def execute(self, statement):
        return _NoConfigResult()


async def test_missing_config_degrades_to_the_full_set():
    declared = await declared_source_types(_NoConfigSession())
    assert declared == list(KNOWN_SOURCE_TYPES)


# <----- Discovery route ----->


@pytest.fixture
async def sources_client():
    app.dependency_overrides[get_declared_source_types] = (
        lambda: ["web", "news"]
    )
    app.dependency_overrides[get_current_user] = lambda: FakeUser(
        id=uuid.uuid4(), roles=[]
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


async def test_sources_route_returns_declared_types(sources_client):
    response = await sources_client.get("/search/sources")
    assert response.status_code == 200
    assert response.json() == {"source_types": ["web", "news"]}
