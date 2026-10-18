"""Offline tests for the Jev-compatible decision client (no network)."""

import json

import httpx
import pytest

from src.modules.decision.base import (
    DecisionError,
    IDecisionMaker,
    temperature_scale,
)
from src.modules.decision.providers.jev import JevCompatibleClient


def _client(handler, **kwargs) -> JevCompatibleClient:
    return JevCompatibleClient(
        base_url="http://laya.test",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _systemone_answer(answer: dict) -> dict:
    return {"model": "laya", "answers": {"q": answer}, "usage": {}}


# <----- temperature scaling ----->


def test_temperature_one_is_identity():
    probs = {"a": 0.7, "b": 0.3}
    assert temperature_scale(probs, 1.0) == probs


def test_temperature_above_one_flattens():
    probs = {"a": 0.9, "b": 0.1}
    scaled = temperature_scale(probs, 2.0)
    assert scaled["a"] < 0.9
    assert abs(sum(scaled.values()) - 1.0) < 1e-9


def test_temperature_below_one_sharpens():
    probs = {"a": 0.7, "b": 0.3}
    scaled = temperature_scale(probs, 0.5)
    assert scaled["a"] > 0.7


def test_invalid_temperature_returns_original():
    probs = {"a": 0.7, "b": 0.3}
    assert temperature_scale(probs, 0.0) == probs


# <----- wire protocol ----->


@pytest.mark.asyncio
async def test_noul_request_shape_and_parsing():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json=_systemone_answer({"type": "noul", "noul": 0.892})
        )

    client = _client(handler, api_key="secret")
    decision = await client.noul("Is it relevant?", "state text")

    assert captured["path"] == "/v1/systemone"
    assert captured["auth"] == "Bearer secret"
    assert captured["body"]["state"] == "state text"
    assert captured["body"]["questions"]["q"]["type"] == "noul"
    assert decision.yes == pytest.approx(0.892)


@pytest.mark.asyncio
async def test_noul_calibration_flattens_overconfidence():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_systemone_answer({"type": "noul", "noul": 0.99})
        )

    client = _client(handler, calibration={"noul:2": 3.0})
    decision = await client.noul("q?", "state")

    assert decision.yes < 0.99
    assert decision.confidence == pytest.approx(max(decision.yes, 1 - decision.yes))


@pytest.mark.asyncio
async def test_score_parses_probabilities_and_legend():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_systemone_answer(
                {
                    "type": "score",
                    "score": 1.6994,
                    "legend": {"0": "calm", "1": "firm", "2": "angry"},
                    "probabilities": {"0": 0.0249, "1": 0.4136, "2": 0.5615},
                }
            ),
        )

    client = _client(handler)
    decision = await client.score("How urgent?", ["calm", "firm", "angry"], "state")

    assert decision.score == pytest.approx(1.6994)
    assert decision.legend["1"] == "firm"
    assert abs(sum(decision.probabilities.values()) - 1.0) < 1e-6
    assert decision.confidence == pytest.approx(max(decision.probabilities.values()))


@pytest.mark.asyncio
async def test_choice_picks_argmax_after_calibration():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_systemone_answer(
                {
                    "type": "choice",
                    "choice": "optimized",
                    "probabilities": {"optimized": 0.60, "raw": 0.40},
                }
            ),
        )

    client = _client(handler, calibration={"choice:2": 1.0})
    decision = await client.choice("Which query?", {"optimized": "o", "raw": "r"}, "s")

    assert decision.choice == "optimized"
    assert decision.probabilities["optimized"] == pytest.approx(0.60)


@pytest.mark.asyncio
async def test_batch_noul_uses_batch_endpoint():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["states"] = json.loads(request.content)["states"]
        return httpx.Response(
            200,
            json={
                "results": [
                    {"q": {"type": "noul", "noul": 0.9}},
                    {"q": {"type": "noul", "noul": 0.2}},
                ]
            },
        )

    client = _client(handler)
    decisions = await client.batch_noul("relevant?", ["s1", "s2"])

    assert captured["path"] == "/v1/systemone/batch"
    assert captured["states"] == ["s1", "s2"]
    assert [d.yes for d in decisions] == pytest.approx([0.9, 0.2])


@pytest.mark.asyncio
async def test_batch_count_mismatch_is_an_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"q": {"noul": 0.5}}]})

    client = _client(handler)
    with pytest.raises(DecisionError, match="malformed"):
        await client.batch_noul("q?", ["s1", "s2"])


@pytest.mark.asyncio
async def test_http_error_becomes_decision_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="overloaded")

    client = _client(handler)
    with pytest.raises(DecisionError, match="503"):
        await client.noul("q?", "state")


@pytest.mark.asyncio
async def test_network_failure_becomes_decision_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = _client(handler)
    with pytest.raises(DecisionError, match="unreachable"):
        await client.noul("q?", "state")


@pytest.mark.asyncio
async def test_missing_answers_is_an_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "laya"})

    client = _client(handler)
    with pytest.raises(DecisionError, match="answers"):
        await client.noul("q?", "state")


# <----- factory ----->


def test_factory_builds_jev_client_from_resolved():
    from src.modules.decision.factory import DecisionFactory
    from src.modules.llm_provider.selection import ResolvedModel

    maker = DecisionFactory().create_decision_maker(
        ResolvedModel(
            kind="decision",
            model_name="english",
            provider_type="jev",
            base_url="http://laya.sidecar:8000",
            api_key="k",
            default_params={
                "calibration": {"noul:2": 1.4, "default": 1.2},
                "timeout": 5.0,
            },
        )
    )

    assert isinstance(maker, JevCompatibleClient)
    assert isinstance(maker, IDecisionMaker)
    assert maker.base_url == "http://laya.sidecar:8000"
    assert maker.api_key == "k"
    assert maker.calibration["noul:2"] == 1.4
    assert maker.timeout == 5.0
    assert maker.model == "english"


def test_factory_rejects_unknown_provider_type():
    from src.modules.decision.factory import DecisionFactory
    from src.modules.llm_provider.selection import ResolvedModel

    with pytest.raises(ValueError, match="provider type"):
        DecisionFactory().create_decision_maker(
            ResolvedModel(
                kind="decision",
                model_name="x",
                provider_type="nope",
                base_url="http://x",
            )
        )
