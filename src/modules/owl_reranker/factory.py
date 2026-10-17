import asyncio
import logging
import uuid

from fastapi import Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from ...tools.db import engine, get_session
from ...tools.settings import settings
from ..llm_provider.model import ModelKind
from ..llm_provider.repository import ModelRepository
from ..llm_provider.selection import ResolvedModel, SelectionError
from .base import IRanker
from .owl_reranker import OwlRanker

logger = logging.getLogger(__name__)


class RankerFactory:
    """Registry of IRanker implementations with instance caching.

    Model selection goes through the catalog (kind='reranker'); the flashrank
    cross-encoder stays the default. Instances load once per model and are
    reused across requests.
    """

    REGISTRY: dict[str, type] = {
        "flashrank": OwlRanker,
    }

    _instances: dict[tuple, IRanker] = {}

    @classmethod
    async def get_or_load(
        cls, provider_type: str, model_name: str, cache_dir: str | None = None
    ) -> IRanker:
        ranker_cls = cls.REGISTRY.get(provider_type)
        if ranker_cls is None:
            raise ValueError(
                f"No ranker implementation for provider type '{provider_type}' "
                f"(known: {sorted(cls.REGISTRY)})."
            )

        key = (provider_type, model_name, cache_dir or "")
        if key not in cls._instances:
            logger.info(
                "Loading ranker model '%s' via %s (first use; may download "
                "weights)...",
                model_name,
                provider_type,
            )
            cls._instances[key] = await asyncio.to_thread(
                ranker_cls, model_name=model_name, cache_dir=cache_dir
            )
        return cls._instances[key]

    @classmethod
    async def from_resolved(cls, resolved: ResolvedModel) -> IRanker:
        cache_dir = resolved.default_params.get("cache_dir")
        return await cls.get_or_load(
            resolved.provider_type, resolved.model_name, cache_dir
        )

    @classmethod
    def reset_cache(cls) -> None:
        """Test hook: drop cached instances."""
        cls._instances = {}


# Reranking is infrastructure, not a user-facing choice yet: the resolver uses
# a synthetic system identity with admin-level roles (role checks still apply
# to the selected entry).
_SYSTEM_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")


async def _resolve_ranker_model(
    session: AsyncSession, requested_model_id: uuid.UUID | None
) -> ResolvedModel:
    repo = ModelRepository(session)
    try:
        return await repo.resolve_model(
            kind=ModelKind.RERANKER.value,
            user_id=_SYSTEM_USER_ID,
            user_roles=["admin"],
            requested_model_id=requested_model_id,
        )
    except SelectionError:
        # Registry has nothing (fresh DB): built-in flashrank default.
        return ResolvedModel(
            kind=ModelKind.RERANKER.value,
            model_name=settings.RANKER_MODEL_NAME,
            provider_type="flashrank",
            default_params={"cache_dir": settings.RANKER_CACHE_DIR},
        )


async def init_ranker() -> IRanker:
    """Preload the default ranker at application startup (never per request)."""
    async with AsyncSession(engine) as session:
        resolved = await _resolve_ranker_model(session, None)

    ranker = await RankerFactory.from_resolved(resolved)
    logger.info("Default ranker ready: %s (%s).", resolved.model_name, resolved.source)
    return ranker


# Dependency
async def get_ranker(
    session: AsyncSession = Depends(get_session),
    request_model_id: uuid.UUID | None = Query(default=None, alias="reranker_id"),
) -> IRanker:
    try:
        resolved = await _resolve_ranker_model(session, request_model_id)
        return await RankerFactory.from_resolved(resolved)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ranker unavailable: {e}",
        ) from e
