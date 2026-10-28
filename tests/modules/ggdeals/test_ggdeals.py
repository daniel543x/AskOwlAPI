"""Offline tests for the gg.deals offers tool (PR 14 / decision 18)."""

import json

import httpx
import pytest

import src.modules.ggdeals.client as ggdeals_module
from src.modules.ggdeals.client import GgDealsClient, parse_offers
from src.modules.ggdeals.factory import reset_client
from src.modules.jobs.gates import ResourceGateRegistry
from src.tools.cache import MemoryCache
from src.tools.settings import settings


@pytest.fixture(autouse=True)
def clean_state():
    ResourceGateRegistry.reset()
    reset_client()
    yield
    ResourceGateRegistry.reset()
    reset_client()


def configure(**overrides):
    defaults = {
        "GGDEALS_ENABLED": True,
        "GGDEALS_API_KEY": "test-key",
        "GGDEALS_PRICES_PATH": "/v1/prices/search",
    }
    defaults.update(overrides)
    for name, value in defaults.items():
        setattr(settings, name, value)


def transport_with(payload: dict | list):
    async def handler(request: httpx.Request) -> httpx.Response:
        assert "key=test-key" in str(request.url)
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


def patch_transport(monkeypatch, payload):
    real = httpx.AsyncClient
    monkeypatch.setattr(
        ggdeals_module.httpx,
        "AsyncClient",
        lambda **kwargs: real(**kwargs, transport=transport_with(payload)),
    )


# <----- Parsing ----->


def test_parse_offers_normalizes_wrapped_payloads():
    offers = parse_offers(
        {
            "data": {
                "results": [
                    {
                        "title": "Baldur's Gate 3",
                        "price": {"amount": 44.99, "store": "Steam"},
                        "historical_low": {"amount": 39.99},
                        "stores": ["Steam", "GOG"],
                    }
                ]
            }
        }
    )
    assert offers[0]["title"] == "Baldur's Gate 3"
    assert offers[0]["current_best"]["amount"] == 44.99
    assert offers[0]["historical_low"]["amount"] == 39.99
    assert offers[0]["stores"] == ["Steam", "GOG"]


def test_parse_offers_drops_unusable_items_and_junk():
    assert parse_offers([{"unrelated": 1}, {"title": "Ok", "price": 5}, "junk"]) == [
        {"title": "Ok", "current_best": 5, "historical_low": None, "stores": None}
    ]
    assert parse_offers({"unexpected": "shape"}) == []


# <----- Client behavior ----->


async def test_disabled_or_unconfigured_client_degrades_to_empty(monkeypatch):
    configure(GGDEALS_ENABLED=False)
    assert await GgDealsClient(MemoryCache()).game_offers("hades") == []

    configure(GGDEALS_PRICES_PATH="")
    assert await GgDealsClient(MemoryCache()).game_offers("hades") == []


async def test_offers_are_cached_and_pass_the_gate(monkeypatch):
    configure()
    calls: list[int] = []

    real = httpx.AsyncClient

    def counting(**kwargs):
        calls.append(1)
        return real(**kwargs, transport=transport_with(
            {"results": [{"title": "Hades", "price": 19.99}]}
        ))

    monkeypatch.setattr(ggdeals_module.httpx, "AsyncClient", counting)
    cache = MemoryCache()
    client = GgDealsClient(cache)

    first = await client.game_offers("hades")
    second = await client.game_offers("hades")

    assert len(calls) == 1  # second lookup is a cache hit
    assert first == second
    assert first[0]["title"] == "Hades"
    assert "ggdeals:offers:hades" in cache._store


async def test_api_failure_degrades_to_empty(monkeypatch):
    configure()
    real = httpx.AsyncClient

    def failing(**kwargs):
        async def handler(request):
            raise httpx.ConnectError("down")

        return real(**kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(ggdeals_module.httpx, "AsyncClient", failing)
    assert await GgDealsClient(MemoryCache()).game_offers("hades") == []


# <----- Conditional tool attach + offers event ----->


def test_tool_absent_when_disabled():
    from src.core.tools.toolbox import AskToolbox, build_tools

    configure(GGDEALS_ENABLED=False)
    toolbox = AskToolbox(None, None, None, None, None)
    assert all(t.name != "game_offers" for t in build_tools(toolbox))


def test_tool_absent_when_unconfigured():
    from src.core.tools.toolbox import AskToolbox, build_tools

    configure(GGDEALS_PRICES_PATH="")
    toolbox = AskToolbox(None, None, None, None, None)
    assert all(t.name != "game_offers" for t in build_tools(toolbox))


def test_tool_present_when_configured():
    from src.core.tools.toolbox import AskToolbox, build_tools

    configure()
    toolbox = AskToolbox(None, None, None, None, None)
    tools = build_tools(toolbox)
    assert [t.name for t in tools if t.name == "game_offers"]


async def test_tool_records_structured_offers(monkeypatch):
    from src.core.tools.toolbox import AskToolbox, build_tools

    configure()
    patch_transport(
        monkeypatch, {"results": [{"title": "Hades", "price": 19.99}]}
    )
    toolbox = AskToolbox(None, None, None, None, None)
    tool = next(t for t in build_tools(toolbox) if t.name == "game_offers")

    result = await tool.coroutine(game_name="Hades")

    parsed = json.loads(result)
    assert parsed["count"] == 1
    assert parsed["offers"][0]["title"] == "Hades"
    assert toolbox.take_new_offers()[0]["title"] == "Hades"
    assert toolbox.take_new_offers() == []  # drained
