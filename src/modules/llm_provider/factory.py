from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from ...tools.cryptography import decrypt
from .model import LLMModelCatalog, LLMProvider


class LLMFactory:
    def create_llm(
        self, model_config: LLMModelCatalog, provider_config: LLMProvider
    ) -> BaseChatModel:

        params = (
            model_config.default_params.copy() if model_config.default_params else {}
        )

        api_key = None
        if provider_config.api_key_encrypted:
            try:
                api_key = decrypt(provider_config.api_key_encrypted)
            except Exception as e:
                print(f"Can't decrypt API KEY: {e}")
                pass

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
            )
