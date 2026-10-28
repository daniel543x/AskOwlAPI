"""FastAPI dependencies resolving models from the registry for routes.

1. Resolve the model (request param -> user preference -> global default ->
   built-in local default) with role checks.
2. Enforce the capabilities the calling feature needs (e.g. `tools`).
3. Validate the reasoning controls against capabilities + effective effort
   levels (422 with a stable error code, decision 11).
4. Build the client for the model's kind and return it ready to use.
"""

import uuid

from fastapi import Depends, HTTPException, Query, status
from langchain_core.language_models import BaseChatModel

from ..identity.auth.security import get_current_user
from ..identity.user.model import User
from .capability_probe import effective_reasoning_state
from .factory import ChatModelFactory, get_llm_factory
from .model import ModelKind
from .reasoning import ReasoningError, adapter_wire_params, validate_reasoning
from .repository import ModelRepository, get_repository
from .selection import ResolvedModel, SelectionError


class MissingCapabilityError(Exception):
    """The resolved model lacks a capability the feature requires."""


def ensure_capability(resolved: ResolvedModel, capability: str) -> None:
    if capability not in resolved.capabilities:
        raise MissingCapabilityError(
            f"Model '{resolved.model_name}' (provider '{resolved.provider_type}') "
            f"lacks the required '{capability}' capability. Select a "
            f"tool-capable model via the model_id parameter or ask the admin "
            f"to mark it with the '{capability}' capability."
        )


def _to_http_error(error: SelectionError) -> HTTPException:
    if error.tier == "permission":
        return HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=error.reason
        )
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=error.reason
    )


def _reasoning_http_error(error: ReasoningError) -> HTTPException:
    """422 with a machine-readable body: {"code", "message"}."""
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail={"code": error.code, "message": error.message},
    )


def _user_role_names(user: User) -> list[str]:
    return [role.name for role in user.roles]


async def build_chat_model(
    repo: ModelRepository,
    factory: ChatModelFactory,
    user: User,
    request_model_id: uuid.UUID | None,
    thinking: bool | None = None,
    effort: str | None = None,
) -> BaseChatModel:
    """Resolve, gate, and build the chat model for one ask turn.

    `thinking`/`effort` come from the request (POST body or GET query).
    Shared by every ask route so the reasoning contract is identical.
    """
    try:
        resolved = await repo.resolve_model(
            kind=ModelKind.CHAT.value,
            user_id=user.id,
            user_roles=_user_role_names(user),
            requested_model_id=request_model_id,
        )
        ensure_capability(resolved, "tools")
    except SelectionError as e:
        raise _to_http_error(e) from e
    except MissingCapabilityError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e)
        ) from e

    try:
        capabilities, effort_levels = await effective_reasoning_state(resolved)
        validate_reasoning(capabilities, effort_levels, thinking, effort)
    except ReasoningError as e:
        raise _reasoning_http_error(e) from e

    return factory.create_chat_model(
        resolved, reasoning_params=adapter_wire_params(resolved, thinking, effort)
    )


async def get_llm_client(
    repo: ModelRepository = Depends(get_repository),
    factory: ChatModelFactory = Depends(get_llm_factory),
    user: User = Depends(get_current_user),
    request_model_id: uuid.UUID | None = Query(default=None, alias="model_id"),
    thinking: bool | None = Query(default=None),
    effort: str | None = Query(default=None),
) -> BaseChatModel:
    """Chat model for GET ask routes; `thinking`/`effort` are query params.

    The ask pipeline talks to the LLM exclusively through native tool calls,
    so models without tool support are rejected up front with a 4xx. POST
    ask routes call `build_chat_model` directly with their body fields.
    """
    return await build_chat_model(
        repo, factory, user, request_model_id, thinking=thinking, effort=effort
    )
