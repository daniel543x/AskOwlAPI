import hashlib
import inspect
import json
import logging
from typing import Any

from fastapi import Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from ...tools.db import get_session
from ...tools.settings import settings
from .caching import CachedSearchProvider
from .model import SearchProviderConfig
from .providers.base import ISearchProvider
from .providers.searxng import SearXNGProvider
from .source_types import KNOWN_SOURCE_TYPES

logger = logging.getLogger(__name__)


def filter_provider_kwargs(
    provider_cls: type[ISearchProvider], settings_dict: dict[str, Any]
) -> dict[str, Any]:
    """Keep only kwargs the provider constructor actually accepts.

    Database settings blobs are free-form JSON; unknown keys are dropped
    (with a debug log) instead of crashing provider construction.
    """
    if not settings_dict:
        return {}

    accepted = set(inspect.signature(provider_cls.__init__).parameters)
    accepted.discard("self")

    filtered = {k: v for k, v in settings_dict.items() if k in accepted}
    dropped = sorted(set(settings_dict) - set(filtered))
    if dropped:
        logger.debug(
            "Ignoring unsupported settings %s for provider '%s'",
            dropped,
            provider_cls.__name__,
        )
    return filtered


class SearchProviderFactory:
    # Registry of available provider types
    _providers: dict[str, type[ISearchProvider]] = {
        "searxng": SearXNGProvider,
    }

    # One shared CachedSearchProvider per provider configuration, so the
    # cache outlives per-request provider construction (WU-10).
    _cache_wrappers: dict[str, CachedSearchProvider] = {}

    @classmethod
    def _cache_key(cls, provider_type: str, settings_to_apply: dict[str, Any]) -> str:
        blob = json.dumps(settings_to_apply, sort_keys=True, default=str)
        digest = hashlib.sha256(blob.encode()).hexdigest()[:16]
        return f"{provider_type}:{digest}"

    @classmethod
    def register_provider(cls, name: str, provider_class: type[ISearchProvider]):
        cls._providers[name] = provider_class

    @classmethod
    async def get_default_provider(
        cls, session: AsyncSession
    ) -> tuple[ISearchProvider, str]:
        """Get the default search provider based on database configuration."""
        provider = await cls.create_provider(session)
        return provider, type(provider).__name__

    @classmethod
    def _provider_for(
        cls, provider_type: str, settings_to_apply: dict[str, Any]
    ) -> ISearchProvider:
        """Constructed provider for one configuration, cache-decorated.

        The factory runs per request, so the wrapper (and its inner
        provider) must live in a module-level registry keyed by the config
        blob — otherwise the cache would silently reset every request.
        """
        if not settings.SEARCH_CACHE:
            provider_cls = cls._providers[provider_type]
            return provider_cls(
                **filter_provider_kwargs(provider_cls, settings_to_apply)
            )

        key = cls._cache_key(provider_type, settings_to_apply)
        wrapper = cls._cache_wrappers.get(key)
        if wrapper is None:
            provider_cls = cls._providers[provider_type]
            provider: ISearchProvider = provider_cls(
                **filter_provider_kwargs(provider_cls, settings_to_apply)
            )
            wrapper = CachedSearchProvider(
                provider,
                ttl_seconds=settings.SEARCH_CACHE_TTL_S,
                max_entries=settings.SEARCH_CACHE_MAX_ENTRIES,
            )
            cls._cache_wrappers[key] = wrapper
        return wrapper

    @classmethod
    async def create_provider(
        cls, session: AsyncSession, provider_type: str | None = None
    ) -> ISearchProvider:
        """
        Create a search provider instance.
        If provider_type is None, uses the default provider from database.
        If provider_type is specified/given, creates that provider type.
        """

        settings_to_apply: dict[str, Any] = {}

        if provider_type:
            statement = select(SearchProviderConfig).where(
                SearchProviderConfig.provider_type == provider_type,
                SearchProviderConfig.is_active,
            )
            result = await session.execute(statement)
            config = result.scalar_one_or_none()

            if config:
                settings_to_apply = config.settings

        else:
            statement = select(SearchProviderConfig).where(
                SearchProviderConfig.is_active
            )
            result = await session.execute(statement)
            config = result.scalar_one_or_none()

            if config:
                provider_type = config.provider_type
                settings_to_apply = config.settings
            else:
                # No database configuration: fall back to environment settings.
                provider_type = "searxng"
                settings_to_apply = {}

        if provider_type not in cls._providers:
            raise ValueError(f"Unsupported provider type: {provider_type}")

        provider = cls._provider_for(provider_type, settings_to_apply)
        # Resource-gate identity (PR 12): the search gate is keyed per
        # provider config; concurrency comes from the settings blob.
        provider.search_resource_id = str(config.id) if config else None
        provider.search_max_concurrency = int(
            (config.settings.get("max_concurrency") if config else None)
            or settings.DEFAULT_PROVIDER_MAX_CONCURRENCY
        )
        return provider


# Dependency
async def get_search_provider(
    session: AsyncSession = Depends(get_session),
    search_provider: str | None = Query(None, alias="search_provider"),
):
    return await SearchProviderFactory.create_provider(session, search_provider)


async def declared_source_types(
    session: AsyncSession,
    provider_type: str | None = None,
) -> list[str]:
    """Source types the active/selected provider declares (PR 11).

    No database configuration degrades to the full v1 set (the built-in
    SearXNG default): a missing config row must not make every request
    fail validation.
    """
    statement = select(SearchProviderConfig).where(SearchProviderConfig.is_active)
    if provider_type:
        statement = statement.where(
            SearchProviderConfig.provider_type == provider_type
        )
    result = await session.execute(statement)
    config = result.scalar_one_or_none()
    if config and config.supported_source_types:
        return list(config.supported_source_types)
    return list(KNOWN_SOURCE_TYPES)


async def get_declared_source_types(
    session: AsyncSession = Depends(get_session),
    search_provider: str | None = Query(None, alias="search_provider"),
) -> list[str]:
    return await declared_source_types(session, search_provider)
