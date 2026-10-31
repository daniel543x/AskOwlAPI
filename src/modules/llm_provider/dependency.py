"""FastAPI dependencies resolving models from the registry for routes.

1. Resolve the model (request param -> user preference -> global default ->
   built-in local default) with role checks.
2. Enforce the capabilities the calling feature needs (e.g. `tools`).
3. Validate the reasoning controls against capabilities + effective effort
   levels (422 with a stable error code, decision 11).
4. Build the client for the model's kind and return it ready to use.
"""

import uuid

from fastapi import Depends, HTTPException, Query, Request, status
from langchain_core.language_models import BaseChatModel

from ...tools.settings import settings
from ..identity.auth.security import get_current_user
from ..identity.user.model import User
from ..jobs.gates import TurnResources, model_gate, search_gate
from ..profile.factory import get_optional_user_preferences
from ..profile.model import UserPreferences
from ..search_engine.factory import get_search_provider
from ..search_engine.providers.base import ISearchProvider
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
    profile=None,
) -> BaseChatModel:
    """Resolve, gate, and build the chat model for one ask turn.

    `thinking`/`effort` come from the request (POST body or GET query).
    `profile` is the caller's preference row (decision 12): its
    default_model_id sits between the explicit request and the per-kind
    preference in the resolution order, and its thinking/effort fill the
    slots the request left unset — soft-clamped to the resolved model
    (profile values never 422; explicit request values do).
    """
    profile_model_id = profile.default_model_id if profile else None
    try:
        resolved = await repo.resolve_model(
            kind=ModelKind.CHAT.value,
            user_id=user.id,
            user_roles=_user_role_names(user),
            requested_model_id=request_model_id,
            profile_model_id=profile_model_id,
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

    if profile is not None:
        # Decision 12: request > profile > model default. Profile values
        # were clamped on save, but the effective model may have changed
        # since — clamp again here, silently.
        if thinking is None and "thinking" in capabilities:
            thinking = profile.thinking
        if effort is None and "effort" in capabilities:
            candidate = profile.effort
            if candidate and (not effort_levels or candidate in effort_levels):
                effort = candidate

    return factory.create_chat_model(
        resolved, reasoning_params=adapter_wire_params(resolved, thinking, effort)
    )


async def get_llm_client(
    http_request: Request,
    repo: ModelRepository = Depends(get_repository),
    factory: ChatModelFactory = Depends(get_llm_factory),
    user: User = Depends(get_current_user),
    request_model_id: uuid.UUID | None = Query(default=None, alias="model_id"),
    thinking: bool | None = Query(default=None),
    effort: str | None = Query(default=None),
    profile: UserPreferences | None = Depends(get_optional_user_preferences),
) -> BaseChatModel:
    """Chat model for every ask route — the single override seam.

    GET ask routes carry `thinking`/`effort` as query params; POST routes
    carry them in the body, read here from the raw JSON (FastAPI caches
    the body, so the route's own typed parsing is unaffected). The ask
    pipeline talks to the LLM exclusively through native tool calls, so
    models without tool support are rejected up front with a 4xx.
    """
    if http_request.method == "POST":
        try:
            body = await http_request.json()
        except Exception:
            body = None
        if isinstance(body, dict):
            if thinking is None:
                thinking = body.get("thinking")
            if effort is None:
                effort = body.get("effort")
    return await build_chat_model(
        repo,
        factory,
        user,
        request_model_id,
        thinking=thinking,
        effort=effort,
        profile=profile,
    )


def turn_resources_for_resolved(
    resolved: ResolvedModel, searching: ISearchProvider
) -> TurnResources:
    """Gate keys + capabilities for one resolved model (PR 12).

    The model gate is keyed per provider; the search gate per provider
    config. Only gate identity is computed here — acquisition happens at
    the turn boundary in the adapters.
    """
    return TurnResources(
        keys=[
            model_gate(resolved.provider_id, resolved.max_concurrency),
            search_gate(
                getattr(searching, "search_resource_id", None),
                getattr(
                    searching,
                    "search_max_concurrency",
                    settings.DEFAULT_PROVIDER_MAX_CONCURRENCY,
                ),
            ),
        ],
        model_capabilities=resolved.capabilities,
    )


async def get_turn_resources(
    repo: ModelRepository = Depends(get_repository),
    user: User = Depends(get_current_user),
    request_model_id: uuid.UUID | None = Query(default=None, alias="model_id"),
    profile: UserPreferences | None = Depends(get_optional_user_preferences),
    searching: ISearchProvider = Depends(get_search_provider),
) -> TurnResources:
    """The gates one ask turn must hold: model + search (PR 12).

    Resolves the same model the LLM dependency resolves (request >
    profile > preference > default) to key the model gate.
    """
    resolved = await repo.resolve_model(
        kind=ModelKind.CHAT.value,
        user_id=user.id,
        user_roles=_user_role_names(user),
        requested_model_id=request_model_id,
        profile_model_id=profile.default_model_id if profile else None,
    )
    return turn_resources_for_resolved(resolved, searching)
