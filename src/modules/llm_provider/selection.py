"""Pure model-selection logic, independent of the database.

Resolution order (identical for every model kind):
1. request param (role-checked)
2. user preference for the kind (role-checked)
3. global default for the kind (role-checked)
4. built-in local default (handled by the repository layer)
"""

import uuid
from dataclasses import dataclass, field
from typing import Any

ROLE_HIERARCHY = {"user": 0, "research": 1, "admin": 2}

# The seeded role was renamed to `research` (PLAN decision #4; the data
# migration renames role rows, not catalog strings), so legacy catalog
# rows may still carry the pre-rename name. It keeps its slot.
_LEGACY_ROLE_ALIASES = {"pro": "research"}

# Tiers in resolution order, used for error reporting.
TIER_REQUEST = "request"
TIER_PROFILE = "profile"
TIER_PREFERENCE = "preference"
TIER_DEFAULT = "default"
TIER_BUILTIN = "builtin"


@dataclass
class ResolvedModel:
    """A model selected from the registry, ready for a factory.

    `api_key` is decrypted; it must never be logged or serialized back out.
    `entry_id` is None for the built-in local default tier.
    """

    kind: str
    model_name: str
    provider_type: str
    base_url: str | None = None
    api_key: str | None = None
    default_params: dict[str, Any] = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)
    effort_levels: list[str] = field(default_factory=list)
    vector_dims: int | None = None
    entry_id: uuid.UUID | None = None
    # Resource-gate identity (PR 12): the model gate is keyed per provider.
    provider_id: uuid.UUID | None = None
    max_concurrency: int = 1
    source: str = TIER_DEFAULT


class SelectionError(Exception):
    """No acceptable model could be selected."""

    def __init__(self, reason: str, tier: str = TIER_DEFAULT):
        super().__init__(reason)
        self.reason = reason
        self.tier = tier


@dataclass(frozen=True)
class ModelInfo:
    """The subset of a catalog entry the selection needs."""

    id: uuid.UUID
    kind: str
    required_role: str
    is_active: bool


@dataclass
class CatalogView:
    """Everything selection needs, pre-fetched from the database."""

    models: dict[uuid.UUID, ModelInfo] = field(default_factory=dict)
    user_preferences: dict[str, uuid.UUID] = field(default_factory=dict)
    defaults: dict[str, uuid.UUID] = field(default_factory=dict)


def _role_level(role: str) -> int | None:
    """Level of a named role; None for unknown names."""
    name = _LEGACY_ROLE_ALIASES.get(role, role)
    return ROLE_HIERARCHY.get(name)


def has_role_access(user_roles: list[str], required_role: str) -> bool:
    """Max-level gate with least-privilege defaults.

    An empty or whitespace-only `required_role` gates nothing; an unknown
    role name denies everyone (a typo in the catalog must not silently
    open the gate), and unknown names in a user's role set contribute
    nothing to their level.
    """
    if not required_role.strip():
        return True
    required_level = _role_level(required_role)
    if required_level is None:
        return False
    user_level = max(
        (
            level
            for role in user_roles
            if (level := _role_level(role)) is not None
        ),
        default=0,
    )
    return user_level >= required_level


def _check_model(
    view: CatalogView, model_id: uuid.UUID, user_roles: list[str]
) -> ModelInfo:
    info = view.models.get(model_id)
    if info is None:
        raise KeyError(model_id)
    if not info.is_active:
        raise SelectionError(f"Model {model_id} is not active.")
    if not has_role_access(user_roles, info.required_role):
        raise SelectionError(
            f"Model {model_id} requires role '{info.required_role}'.",
            tier="permission",
        )
    return info


def _model_of_kind(info: ModelInfo, kind: str) -> bool:
    return info.kind == kind


def resolve_model_id(
    kind: str,
    requested_model_id: uuid.UUID | None,
    view: CatalogView,
    user_roles: list[str],
    profile_model_id: uuid.UUID | None = None,
) -> tuple[uuid.UUID, str]:
    """Return (model_id, tier) following the resolution order.

    The profile default (decision 12) sits between the explicit request
    and the per-kind user preference. A stale profile reference (model
    gone from the catalog) degrades to the next tier. Raises
    SelectionError with the tier that failed, or KeyError when a
    requested/preferred/default model id does not exist in the catalog.
    """
    if requested_model_id is not None:
        info = _check_model(view, requested_model_id, user_roles)
        if not _model_of_kind(info, kind):
            raise SelectionError(
                f"Model {requested_model_id} is of kind '{info.kind}', "
                f"not '{kind}'.",
                tier=TIER_REQUEST,
            )
        return requested_model_id, TIER_REQUEST

    if profile_model_id is not None:
        try:
            info = _check_model(view, profile_model_id, user_roles)
            if _model_of_kind(info, kind):
                return profile_model_id, TIER_PROFILE
        except KeyError:
            pass  # stale profile reference; keep resolving down the chain

    preference_id = view.user_preferences.get(kind)
    if preference_id is not None:
        info = _check_model(view, preference_id, user_roles)
        if _model_of_kind(info, kind):
            return preference_id, TIER_PREFERENCE

    default_id = view.defaults.get(kind)
    if default_id is not None:
        info = _check_model(view, default_id, user_roles)
        if _model_of_kind(info, kind):
            return default_id, TIER_DEFAULT

    raise SelectionError(
        f"No configured {kind} model passed the resolution order "
        "(request param -> user preference -> global default).",
        tier=TIER_BUILTIN,
    )
