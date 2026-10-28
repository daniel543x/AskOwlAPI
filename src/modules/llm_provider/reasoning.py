"""Reasoning controls: neutral `thinking`/`effort` request fields, per-provider
wire adapters (decisions 11 + 20).

The API never speaks provider dialects: `thinking: bool | null` and
`effort: "low"|"medium"|"high"|"max" | null` are the only public spelling.
`reasoning_effort` (OpenAI) or `reasoning` (Ollama) exist only inside this
module's adapters. Gating is capability-based (422), mirroring the `tools`
gate; an effort level must also be in the entry's effective effort levels
(admin override > probe > seed, see capability_probe).

Fallback: the Ollama level probe is the server's 400 rejection — when a
constructed Ollama client rejects the level value up front (pydantic), the
adapter falls back to a boolean and reports the level as unsupported so the
cached level list can be pruned.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessageChunk

from .model import EFFORT_LEVELS
from .selection import ResolvedModel

logger = logging.getLogger(__name__)

# Error codes surfaced as 422 details ({"code": ..., "message": ...}); the
# webapp renders them, so they are part of the protocol.
CODE_MODEL_LACKS_THINKING = "model_lacks_thinking"
CODE_MODEL_LACKS_EFFORT = "model_lacks_effort"
CODE_UNSUPPORTED_EFFORT_LEVEL = "unsupported_effort_level"


class ReasoningError(Exception):
    """A requested thinking/effort combination the model does not support."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def validate_reasoning(
    capabilities: list[str],
    effort_levels: list[str],
    thinking: bool | None,
    effort: str | None,
) -> None:
    """Raise `ReasoningError` for impossible combinations (422 mapping)."""
    if thinking is not None and "thinking" not in capabilities:
        raise ReasoningError(
            CODE_MODEL_LACKS_THINKING,
            "Model does not support the 'thinking' control; omit it or pick "
            "a model with the 'thinking' capability.",
        )
    if effort is None:
        return
    if "effort" not in capabilities:
        raise ReasoningError(
            CODE_MODEL_LACKS_EFFORT,
            "Model does not support the 'effort' control; omit it or pick "
            "a model with the 'effort' capability.",
        )
    if effort not in EFFORT_LEVELS:
        raise ReasoningError(
            CODE_UNSUPPORTED_EFFORT_LEVEL,
            f"Unknown effort level '{effort}'; "
            f"valid values: {list(EFFORT_LEVELS)}.",
        )
    if effort_levels and effort not in effort_levels:
        raise ReasoningError(
            CODE_UNSUPPORTED_EFFORT_LEVEL,
            f"Model does not support effort level '{effort}'; "
            f"supported levels: {effort_levels}.",
        )


def adapter_wire_params(
    resolved: ResolvedModel, thinking: bool | None, effort: str | None
) -> dict[str, Any]:
    """Translate the neutral fields into the provider's wire parameters.

    Returns construction kwargs merged over the catalog `default_params`
    (request wins). Providers without a runtime toggle return `{}`.
    """
    provider = resolved.provider_type.lower()
    if thinking is None and effort is None:
        return {}

    if provider == "ollama":
        # langchain-ollama exposes the wire `think` control as `reasoning`
        # (bool or level string), sent per request. An effort level implies
        # thinking is enabled.
        return {"reasoning": effort if effort is not None else thinking}

    if provider == "openai":
        # `reasoning_effort` is OpenAI's wire spelling; there is no boolean
        # toggle on the models that support it (reasoning is always on).
        if effort is not None:
            return {"reasoning_effort": effort}
        logger.debug(
            "Boolean 'thinking' has no OpenAI wire spelling; ignored for %s.",
            resolved.model_name,
        )
        return {}

    if provider == "openrouter":
        reasoning: dict[str, Any] = {}
        if thinking is not None:
            reasoning["enabled"] = thinking
        if effort is not None:
            reasoning["effort"] = effort
        return {"reasoning": reasoning}

    if provider == "zai":
        # z.ai native fields over the OpenAI-compatible endpoint: GLM <= 5.2
        # accepts {type: disabled}; GLM-5.3+ cannot disable thinking, which
        # the catalog expresses by omitting the 'thinking' capability (the
        # gate above rejects the request before we get here).
        body: dict[str, Any] = {}
        if thinking is not None:
            body["thinking"] = {"type": "enabled" if thinking else "disabled"}
        if effort is not None:
            body["reasoning_effort"] = effort
        return {"extra_body": body}

    # deepseek: no runtime toggle (two catalog entries: chat / reasoner).
    # Unknown provider types: degrade silently — a control the client
    # cannot spell must never break a request.
    logger.debug(
        "Provider type '%s' has no reasoning adapter; controls ignored.",
        resolved.provider_type,
    )
    return {}


def extract_reasoning_delta(chunk: Any) -> str:
    """Reasoning-trace text streamed in one chunk (empty when none).

    Both langchain-ollama (`reasoning=True`) and OpenAI-compatible
    reasoning models surface the trace as `additional_kwargs[
    'reasoning_content']`; that one convention is all the passthrough needs.
    """
    if not isinstance(chunk, AIMessageChunk):
        return ""
    content = chunk.additional_kwargs.get("reasoning_content")
    return content if isinstance(content, str) else ""
