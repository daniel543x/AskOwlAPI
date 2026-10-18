"""Decision-maker construction from registry-resolved catalog entries.

`decision`-kind entries point at a Jev-compatible endpoint (provider type
`jev`): a `laya[serve]` sidecar, a host/LAN service, or the Jev cloud.
Calibration temperatures live in the entry's default_params, e.g.
{"calibration": {"noul:2": 1.4, "score:5": 1.8, "default": 1.2}} —
fitted on your own data per the Laya docs before trusting probabilities.
"""

import logging
import uuid

from fastapi import Depends, Query
from langchain_core.language_models import BaseChatModel  # noqa: F401  (type hints)

from ..identity.auth.security import get_current_user
from ..identity.user.model import User
from ..llm_provider.model import ModelKind
from ..llm_provider.repository import ModelRepository, get_repository
from ..llm_provider.selection import ResolvedModel, SelectionError
from .base import DecisionError, IDecisionMaker
from .providers.jev import JevCompatibleClient

logger = logging.getLogger(__name__)


class DecisionFactory:
    """Builds decision makers from registry-resolved configuration."""

    REGISTRY: dict[str, type] = {
        "jev": JevCompatibleClient,
    }

    def create_decision_maker(self, resolved: ResolvedModel) -> IDecisionMaker:
        client_cls = self.REGISTRY.get(resolved.provider_type)
        if client_cls is None:
            raise ValueError(
                f"No decision implementation for provider type "
                f"'{resolved.provider_type}' (known: {sorted(self.REGISTRY)})."
            )
        if not resolved.base_url:
            raise ValueError("Decision entries need a provider base_url.")

        params = resolved.default_params or {}
        calibration = params.get("calibration") or {}
        if not isinstance(calibration, dict):
            raise ValueError("Entry default_params.calibration must be an object.")

        return client_cls(
            base_url=resolved.base_url,
            api_key=resolved.api_key,
            timeout=float(params.get("timeout", 30.0)),
            calibration={str(k): float(v) for k, v in calibration.items()},
            model=resolved.model_name,
        )


def get_decision_factory() -> DecisionFactory:
    return DecisionFactory()


# Dependency: optional decision maker for pre-scores. Degrades to None when
# nothing is configured instead of failing the request (PLAN Phase 7: the
# whole decision layer must be optional and off the hot path by default).
async def get_optional_decision_maker(
    repo: ModelRepository = Depends(get_repository),
    factory: DecisionFactory = Depends(get_decision_factory),
    user: User = Depends(get_current_user),
    request_model_id: uuid.UUID | None = Query(
        default=None, alias="decision_model_id"
    ),
) -> IDecisionMaker | None:
    try:
        resolved = await repo.resolve_model(
            kind=ModelKind.DECISION.value,
            user_id=user.id,
            user_roles=[role.name for role in user.roles],
            requested_model_id=request_model_id,
        )
        return factory.create_decision_maker(resolved)
    except (SelectionError, ValueError, DecisionError) as e:
        logger.info(
            "Decision pre-scores disabled for this request (no usable "
            "decision model): %s",
            e,
        )
        return None
