import logging

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from ...tools.cryptography import get_crypto_service
from .model import LLMModelCatalog, LLMProvider

logger = logging.getLogger(__name__)


class LLMFactory:
    def __init__(self) -> None:
        self.crypto_service = get_crypto_service()

    def create_llm(
        self, model_config: LLMModelCatalog, provider_config: LLMProvider
    ) -> BaseChatModel:

        params = (
            model_config.default_params.copy() if model_config.default_params else {}
        )

        api_key = None
        if provider_config.api_key_encrypted:
            try:
                api_key = self.crypto_service.decrypt(
                    provider_config.api_key_encrypted
                )
            except Exception:
                # Never log the key or the decrypted/encrypted material.
                logger.exception(
                    "Failed to decrypt API key for provider '%s'; continuing.",
                    provider_config.name,
                )

        if provider_config.base_url:
            params["base_url"] = provider_config.base_url

        if api_key:
            params["api_key"] = api_key

        provider_type = provider_config.provider_type

        """
        num_ctx
        if provider_type == "ollama":
        """

        try:
            llm = init_chat_model(
                model=model_config.model_name, model_provider=provider_type, **params
            )
            return llm
        except Exception as e:
            raise RuntimeError(
                f"Can't init model: {model_config.model_name} "
                f"for provider: {provider_type}. Error: {e}"
            ) from e


# Deppendency
def get_llm_factory() -> LLMFactory:
    return LLMFactory()
