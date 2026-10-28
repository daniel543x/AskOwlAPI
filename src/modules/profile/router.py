"""Profile preference routes: GET/PUT /profile/preferences (PR 10).

PUT validates against the catalog and clamps unsupported values, storing
and returning the normalized payload + warnings (decision 12: degrade,
never crash). GET returns `{stored, resolved, warnings}`; `?model_id=X`
previews the profile as it would resolve against model X — the "model
switch resets thinking/effort" contract.
"""

import logging
import uuid

from fastapi import APIRouter, Depends, Query

from ...modules.identity.auth.security import get_current_user
from ...modules.identity.user.model import User
from ...modules.llm_provider.capability_probe import effective_reasoning_state
from ...modules.llm_provider.repository import ModelRepository
from ...modules.llm_provider.repository import (
    get_repository as get_model_repository,
)
from ...modules.llm_provider.selection import ResolvedModel, SelectionError
from .factory import get_repository as get_prefs_repository
from .model import (
    PreferencesResolved,
    PreferencesStored,
    PreferencesUpdate,
    PreferencesView,
    UserPreferences,
)
from .repository import PreferencesRepository
from .service import clamp_to_model, validate_sources

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/profile", tags=["Profile"])


async def _effective_model(
    model_repo: ModelRepository,
    user: User,
    model_id: uuid.UUID | None,
) -> ResolvedModel | None:
    """The chat model preferences are clamped against; None when the id is
    unknown, inactive, or not available to the user (degrade, never crash)."""
    role_names = [role.name for role in user.roles]
    try:
        return await model_repo.resolve_model(
            kind="chat",
            user_id=user.id,
            user_roles=role_names,
            requested_model_id=model_id,
        )
    except SelectionError as e:
        logger.debug("Preference model %s did not resolve: %s", model_id, e.reason)
        return None
    except KeyError:
        return None


async def build_view(
    stored: UserPreferences, resolved: ResolvedModel | None
) -> PreferencesView:
    """`stored` + `resolved` views; resolved clamps to the effective model."""
    warnings: list[str] = []
    capabilities: list[str] = []
    levels: list[str] = []
    if resolved is not None:
        capabilities, levels = await effective_reasoning_state(resolved)

    default_model_id = stored.default_model_id
    if default_model_id is not None and resolved is None:
        warnings.append(
            "Cleared default_model_id: the model is unknown, inactive, or "
            "not available to you."
        )
        default_model_id = None

    thinking, effort, clamp_warnings = clamp_to_model(
        capabilities, levels, stored.thinking, stored.effort, default_model_id
    )
    warnings.extend(clamp_warnings)

    return PreferencesView(
        stored=PreferencesStored(
            default_model_id=stored.default_model_id,
            default_mode=stored.default_mode,
            thinking=stored.thinking,
            effort=stored.effort,
            source_types=list(stored.source_types or []),
            updated_at=stored.updated_at,
        ),
        resolved=PreferencesResolved(
            default_model_id=default_model_id,
            default_mode=stored.default_mode,
            thinking=thinking,
            effort=effort,
            source_types=list(stored.source_types or []),
        ),
        warnings=warnings,
    )


@router.get("/preferences", response_model=PreferencesView)
async def get_preferences(
    model_repo: ModelRepository = Depends(get_model_repository),
    prefs_repo: PreferencesRepository = Depends(get_prefs_repository),
    user: User = Depends(get_current_user),
    model_id: uuid.UUID | None = Query(default=None),
):
    """The caller's preferences; `?model_id=X` previews a model switch."""
    stored = await prefs_repo.get(user.id) or UserPreferences(user_id=user.id)
    effective_id = model_id or stored.default_model_id
    resolved = await _effective_model(model_repo, user, effective_id)
    return await build_view(stored, resolved)


@router.put("/preferences", response_model=PreferencesView)
async def put_preferences(
    data: PreferencesUpdate,
    model_repo: ModelRepository = Depends(get_model_repository),
    prefs_repo: PreferencesRepository = Depends(get_prefs_repository),
    user: User = Depends(get_current_user),
):
    """Validate against the catalog, clamp on save, return normalized."""
    warnings: list[str] = []
    source_types, source_warnings = validate_sources(data.source_types or [])
    warnings.extend(source_warnings)

    default_model_id = data.default_model_id
    resolved = await _effective_model(model_repo, user, default_model_id)
    if default_model_id is not None and resolved is None:
        warnings.append(
            "Cleared default_model_id: the model is unknown, inactive, or "
            "not available to you."
        )
        default_model_id = None
    if resolved is None:
        # No stored model: clamp against the user's effective chat model so
        # thinking/effort still respect whatever asks would actually use.
        resolved = await _effective_model(model_repo, user, None)

    capabilities: list[str] = []
    levels: list[str] = []
    if resolved is not None:
        capabilities, levels = await effective_reasoning_state(resolved)
    thinking, effort, clamp_warnings = clamp_to_model(
        capabilities, levels, data.thinking, data.effort, default_model_id
    )
    warnings.extend(clamp_warnings)

    saved = await prefs_repo.save(
        user.id,
        PreferencesUpdate(
            default_model_id=default_model_id,
            default_mode=data.default_mode,
            thinking=thinking,
            effort=effort,
            source_types=source_types,
        ),
    )
    view = await build_view(saved, resolved)
    view.warnings = warnings
    return view
