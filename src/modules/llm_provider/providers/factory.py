import json
from typing import Any, Dict, Optional, Type

from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlmodel import select

from src.tools.cryptography import decrypt
from src.tools.db import get_session

from ..model import LLMConfig, LLMProvider
from .base import LLMProviderBase
from .ollama import OllamaProvider


class LLMClientFactory:
    _registry: Dict[str, Type[LLMProviderBase]] = {
        "ollama": OllamaProvider,
    }

    def __init__(self, session: AsyncSession):
        self.session = session

    def _safe_parse_json(self, json_str: Optional[str]) -> Dict[str, Any]:
        if not json_str:
            return {}
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {}

    async def _get_config_data(
        self, config_name: Optional[str]
    ) -> tuple[LLMConfig, LLMProvider]:
        if config_name:
            statement = select(LLMConfig).where(LLMConfig.config_name == config_name)
        else:
            statement = select(LLMConfig).where(
                LLMConfig.is_default, LLMConfig.is_active
            )

        result = await self.session.execute(statement)
        config = result.scalars().first()

        if not config:
            if config_name:
                raise HTTPException(
                    status_code=404, detail=f"Config '{config_name}' not found."
                )
            else:
                raise HTTPException(
                    status_code=500, detail="No default LLM configuration found."
                )

        provider_statement = select(LLMProvider).where(
            LLMProvider.id == config.provider_id
        )
        provider_result = await self.session.execute(provider_statement)

        provider = provider_result.scalars().first()

        if not provider:
            raise HTTPException(
                status_code=500,
                detail=f"Referenced Provider (ID: {config.provider_id}) not found in database.",
            )

        return config, provider

    async def create_client(self, config_name: Optional[str] = None) -> LLMProviderBase:
        config, provider = await self._get_config_data(config_name)

        provider_params = self._safe_parse_json(provider.extra_connection_params)
        model_params = self._safe_parse_json(config.extra_model_params)

        final_params = {**provider_params, **model_params}

        if "base_url" not in final_params and provider.base_url:
            final_params["base_url"] = provider.base_url

        if provider.api_key_encrypted:
            try:
                decrypted_key = decrypt(provider.api_key_encrypted)
                final_params["api_key"] = decrypted_key
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to decrypt API key for provider '{provider.name}'. Check MASTER_KEY. Error: {str(e)}",
                )

        final_extra_params = json.dumps(final_params)

        provider_class = self._registry.get(provider.provider_type)

        if not provider_class:
            raise ValueError(
                f"Provider type '{provider.provider_type}' is not registered. "
                f"Available: {list(self._registry.keys())}"
            )

        # Wszystkie klasy w rejestrze muszą przyjmować te same argumenty bazowe
        return provider_class(
            model_name=config.model_name, extra_params=final_extra_params
        )


# DEPENDENCY
async def get_llm_client(
    config_name: Optional[str] = None, session: AsyncSession = Depends(get_session)
) -> LLMProviderBase:
    factory = LLMClientFactory(session)
    return await factory.create_client(config_name)
