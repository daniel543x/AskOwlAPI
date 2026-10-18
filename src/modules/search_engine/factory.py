import inspect
import logging
from typing import Any

from fastapi import Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from ...tools.db import get_session
from .model import SearchProviderConfig
from .providers.base import ISearchProvider
from .providers.searxng import SearXNGProvider

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

        provider_cls = cls._providers[provider_type]

        return provider_cls(**filter_provider_kwargs(provider_cls, settings_to_apply))


# Dependency
async def get_search_provider(
    session: AsyncSession = Depends(get_session),
    search_provider: str | None = Query(None, alias="search_provider"),
):
    return await SearchProviderFactory.create_provider(session, search_provider)
