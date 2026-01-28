from typing import Dict, Optional, Type

from fastapi import Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from ...tools.db import get_session
from .model import SearchProviderConfig
from .providers.base import ISearchProvider
from .providers.searxng import SearXNGProvider


class SearchProviderFactory:
    # Registry of available provider types
    _providers: Dict[str, Type[ISearchProvider]] = {
        "searxng": SearXNGProvider,
        # Add more
    }

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[ISearchProvider]):
        cls._providers[name] = provider_class

    @classmethod
    async def get_default_provider(
        cls, session: AsyncSession
    ) -> tuple[ISearchProvider, str]:
        # Get the default search provider based on database configuration.
        # Get the first active provider from the database
        from sqlmodel import select

        statement = select(SearchProviderConfig).where(SearchProviderConfig.is_active)
        result = await session.execute(statement)
        config = result.scalar_one_or_none()

        if config:
            provider_type = config.provider_type
            if provider_type in cls._providers:
                provider = cls._providers[provider_type]()
                # Apply settings from database config if needed
                # For now, we're just returning the provider instance
                return provider, provider_type
            else:
                raise ValueError(f"Unsupported provider type: {provider_type}")
        else:
            # Return default provider if in config not found
            default_provider = SearXNGProvider()
            return default_provider, "searxng"

    @classmethod
    async def create_provider(
        cls, session: AsyncSession, provider_type: Optional[str] = None
    ) -> ISearchProvider:
        """
        Create a search provider instance.
        If provider_type is None, uses the default provider from database.
        If provider_type is specified/given, creates that provider type.
        """

        settings_to_apply = {}

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
                provider_type = "searxng"
                settings_to_apply = {}

        if provider_type not in cls._providers:
            raise ValueError(f"Unsupported provider type: {provider_type}")

        provider_cls = cls._providers[provider_type]

        return provider_cls(**settings_to_apply)


# Dependency
async def get_search_provider(
    session: AsyncSession = Depends(get_session),
    search_provider: Optional[str] = Query(None, alias="search_provider"),
):
    return await SearchProviderFactory.create_provider(session, search_provider)
