"""Persona chat-model resolution (webapp API-NOTES #5).

The persona's model always wins over the request tier when it is usable.
An unusable persona model ends the turn with a `model_confirm` payload
(the SSE event, or the 409 detail for JSON/background callers) unless
the caller consented to the standard resolution chain via
`allow_model_fallback`.
"""

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel

from ....modules.identity.user.model import User
from ....modules.jobs.gates import TurnResources
from ....modules.llm_provider.dependency import (
    build_chat_model,
    turn_resources_for_resolved,
)
from ....modules.llm_provider.factory import ChatModelFactory
from ....modules.llm_provider.model import ModelKind
from ....modules.llm_provider.repository import ModelRepository
from ....modules.llm_provider.selection import SelectionError, unusable_reason

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfirm:
    """Why the persona's model cannot run, plus an advisory fallback id."""

    persona_model: str
    reason: str
    fallback_model: str | None

    def payload(self) -> dict[str, Any]:
        return {
            "persona_model": self.persona_model,
            "reason": self.reason,
            "fallback_model": self.fallback_model,
        }


async def _fallback_model_id(
    repo: ModelRepository,
    user: User,
    roles: list[str],
    profile,
) -> str | None:
    """Advisory display hint for the confirm event; None = built-in tier."""
    try:
        resolved = await repo.resolve_model(
            kind=ModelKind.CHAT.value,
            user_id=user.id,
            user_roles=roles,
            requested_model_id=None,
            profile_model_id=profile.default_model_id if profile else None,
        )
    except SelectionError:
        return None
    return str(resolved.entry_id) if resolved.entry_id is not None else None


async def resolve_persona_model(
    persona,
    allow_model_fallback: bool,
    *,
    fallback_model: BaseChatModel,
    fallback_turn: TurnResources,
    repo: ModelRepository,
    factory: ChatModelFactory,
    user: User,
    profile,
    thinking: bool | None,
    effort: str | None,
    searching,
) -> tuple[BaseChatModel, TurnResources] | ModelConfirm:
    """Decide which chat model runs a persona turn (decision table in
    docs/superpowers/plans/2026-10-31-persona-api-batch.md, Task 4)."""
    roles = [role.name for role in user.roles]
    if persona.model is None:
        return fallback_model, fallback_turn
    state = await repo.entry_state(persona.model)
    reason = unusable_reason(state, roles)
    if reason is not None:
        if not allow_model_fallback:
            return ModelConfirm(
                persona_model=str(persona.model),
                reason=reason,
                fallback_model=await _fallback_model_id(repo, user, roles, profile),
            )
        return fallback_model, fallback_turn
    model = await build_chat_model(
        repo,
        factory,
        user,
        request_model_id=persona.model,
        thinking=thinking,
        effort=effort,
        profile=profile,
    )
    resolved = await repo.resolve_model(
        kind=ModelKind.CHAT.value,
        user_id=user.id,
        user_roles=roles,
        requested_model_id=persona.model,
        profile_model_id=profile.default_model_id if profile else None,
    )
    return model, turn_resources_for_resolved(resolved, searching)
