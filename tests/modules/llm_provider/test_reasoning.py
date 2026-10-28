"""Offline tests for the reasoning contract (decisions 11 + 20)."""

import uuid

import pytest
from langchain_core.messages import AIMessageChunk

from src.modules.llm_provider.model import ModelEntryCreate, ModelKind
from src.modules.llm_provider.reasoning import (
    CODE_MODEL_LACKS_EFFORT,
    CODE_MODEL_LACKS_THINKING,
    CODE_UNSUPPORTED_EFFORT_LEVEL,
    ReasoningError,
    adapter_wire_params,
    extract_reasoning_delta,
    validate_reasoning,
)
from src.modules.llm_provider.selection import ResolvedModel


def resolved(
    provider_type: str, capabilities: list[str], levels: list[str] | None = None
):
    return ResolvedModel(
        kind="chat",
        model_name="m",
        provider_type=provider_type,
        capabilities=capabilities,
        effort_levels=levels or [],
    )


# <----- Gating ----->


def test_thinking_requires_capability():
    with pytest.raises(ReasoningError) as exc_info:
        validate_reasoning(["tools"], [], True, None)
    assert exc_info.value.code == CODE_MODEL_LACKS_THINKING


def test_effort_requires_capability():
    with pytest.raises(ReasoningError) as exc_info:
        validate_reasoning(["tools"], [], None, "high")
    assert exc_info.value.code == CODE_MODEL_LACKS_EFFORT


def test_unknown_effort_value_rejected():
    with pytest.raises(ReasoningError) as exc_info:
        validate_reasoning(["effort"], ["low"], None, "turbo")
    assert exc_info.value.code == CODE_UNSUPPORTED_EFFORT_LEVEL


def test_level_outside_entry_levels_rejected():
    with pytest.raises(ReasoningError) as exc_info:
        validate_reasoning(["effort"], ["low", "medium"], None, "max")
    assert exc_info.value.code == CODE_UNSUPPORTED_EFFORT_LEVEL


def test_null_controls_always_pass():
    validate_reasoning([], [], None, None)


def test_supported_combination_passes():
    validate_reasoning(["tools", "thinking", "effort"], ["low", "max"], False, "max")


# <----- Provider adapters ----->


def test_no_controls_no_wire_params():
    assert adapter_wire_params(resolved("ollama", ["tools"]), None, None) == {}


def test_ollama_bool_and_level():
    assert adapter_wire_params(resolved("ollama", []), True, None) == {
        "reasoning": True
    }
    assert adapter_wire_params(resolved("ollama", []), False, None) == {
        "reasoning": False
    }
    assert adapter_wire_params(resolved("ollama", []), None, "high") == {
        "reasoning": "high"
    }


def test_openai_uses_reasoning_effort_wire_name():
    params = adapter_wire_params(resolved("openai", []), None, "medium")
    assert params == {"reasoning_effort": "medium"}
    # Boolean thinking has no OpenAI spelling: ignored, never breaks.
    assert adapter_wire_params(resolved("openai", []), True, None) == {}


def test_openrouter_reasoning_object():
    params = adapter_wire_params(resolved("openrouter", []), True, "low")
    assert params == {"reasoning": {"enabled": True, "effort": "low"}}


def test_zai_native_mapping():
    params = adapter_wire_params(resolved("zai", []), False, "high")
    assert params == {
        "extra_body": {
            "thinking": {"type": "disabled"},
            "reasoning_effort": "high",
        }
    }


def test_deepseek_and_unknown_degrade_to_noop():
    assert adapter_wire_params(resolved("deepseek", []), True, "low") == {}
    assert adapter_wire_params(resolved("mystery", []), True, "low") == {}


# <----- Reasoning delta extraction ----->


def test_extract_reasoning_delta_reads_convention():
    chunk = AIMessageChunk(content="", additional_kwargs={"reasoning_content": "hmm"})
    assert extract_reasoning_delta(chunk) == "hmm"
    assert extract_reasoning_delta(AIMessageChunk(content="text")) == ""
    assert extract_reasoning_delta("not a chunk") == ""


# <----- Catalog schema ----->


_PROVIDER_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")


def test_catalog_rejects_unknown_effort_levels():
    with pytest.raises(ValueError, match="Unknown effort levels"):
        ModelEntryCreate(
            provider_id=_PROVIDER_ID,
            kind=ModelKind.CHAT,
            model_name="m",
            effort_levels=["turbo"],
        )


def test_catalog_accepts_valid_effort_levels():
    entry = ModelEntryCreate(
        provider_id=_PROVIDER_ID,
        kind=ModelKind.CHAT,
        model_name="m",
        effort_levels=["low", "medium", "high", "max"],
    )
    assert entry.effort_levels == ["low", "medium", "high", "max"]
