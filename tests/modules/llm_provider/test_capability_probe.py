"""Offline tests for capability probing (precedence: admin > probe > seed)."""

import uuid

import httpx
import pytest

import src.modules.llm_provider.capability_probe as probe
from src.modules.llm_provider.capability_probe import effective_reasoning_state
from src.modules.llm_provider.selection import ResolvedModel
from src.tools.cache import MemoryCache


def resolved(
    provider_type="ollama",
    capabilities=(),
    levels=(),
    base_url="http://ollama:11434",
    model_name="qwen3.5",
    entry_id="11111111-1111-1111-1111-111111111111",
):
    return ResolvedModel(
        kind="chat",
        model_name=model_name,
        provider_type=provider_type,
        base_url=base_url,
        capabilities=list(capabilities),
        effort_levels=list(levels),
        entry_id=uuid.UUID(entry_id) if entry_id else None,
    )


def ollama_transport(capabilities: list[str]):
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/api/show")
        return httpx.Response(200, json={"capabilities": capabilities})

    return httpx.MockTransport(handler)


@pytest.fixture
def probe_on(monkeypatch):
    monkeypatch.setattr(probe.settings, "CAPABILITY_PROBE_TTL_S", 300.0)


async def test_admin_levels_win_untouched(monkeypatch):
    monkeypatch.setattr(probe.settings, "CAPABILITY_PROBE_TTL_S", 0.0)
    caps, levels = await effective_reasoning_state(
        resolved(levels=["low"]), MemoryCache()
    )
    assert levels == ["low"]


async def test_ollama_probe_fills_capabilities_and_is_cached(monkeypatch, probe_on):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        probe.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(
            **kwargs, transport=ollama_transport(["completion", "tools", "thinking"])
        ),
    )
    cache = MemoryCache()
    caps, levels = await effective_reasoning_state(
        resolved(capabilities=[], levels=[]), cache
    )

    assert "tools" in caps and "thinking" in caps
    assert levels == []  # Ollama levels are request-level; nothing seeded
    assert probe.probe_cache_key(str((resolved()).entry_id)) in cache._store


async def test_probe_failure_keeps_the_seed(monkeypatch, probe_on):
    real_client = httpx.AsyncClient

    def failing(**kwargs):
        async def handler(request):
            raise httpx.ConnectError("down")

        return real_client(**kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(probe.httpx, "AsyncClient", failing)
    caps, levels = await effective_reasoning_state(
        resolved(provider_type="zai", capabilities=[], levels=[]), MemoryCache()
    )
    # z.ai seed levels survive (degrade, never crash).
    assert levels == ["low", "medium", "high", "max"]
    assert caps == []


async def test_zai_probe_prunes_rejected_levels(monkeypatch, probe_on):
    rejected = {"low"}

    async def handler(request: httpx.Request) -> httpx.Response:
        import json

        level = json.loads(request.content.decode())["reasoning_effort"]
        return httpx.Response(400 if level in rejected else 200, json={})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        probe.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(**kwargs, transport=httpx.MockTransport(handler)),
    )
    _, levels = await effective_reasoning_state(
        resolved(provider_type="zai", capabilities=[], levels=[]), MemoryCache()
    )
    assert levels == ["medium", "high", "max"]


async def test_nothing_to_learn_skips_the_probe(monkeypatch, probe_on):
    def explode(**kwargs):
        raise AssertionError("probe must not run")

    monkeypatch.setattr(probe.httpx, "AsyncClient", explode)
    caps, levels = await effective_reasoning_state(
        resolved(provider_type="openai", capabilities=["tools", "effort"], levels=[]),
        MemoryCache(),
    )
    assert caps == ["tools", "effort"]
    assert levels == []
