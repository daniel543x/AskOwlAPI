import logging

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from .selection import ResolvedModel

logger = logging.getLogger(__name__)


class ChatModelFactory:
    """Builds chat models from registry-resolved configuration."""

    def create_chat_model(self, resolved: ResolvedModel) -> BaseChatModel:
        params = dict(resolved.default_params)

        if resolved.base_url:
            params["base_url"] = resolved.base_url
        if resolved.api_key:
            params["api_key"] = resolved.api_key

        try:
            return init_chat_model(
                model=resolved.model_name,
                model_provider=resolved.provider_type,
                **params,
            )
        except Exception as e:
            # Never include the API key in logs or errors.
            raise RuntimeError(
                f"Can't init chat model '{resolved.model_name}' for provider "
                f"type '{resolved.provider_type}': {e}"
            ) from e


# Dependency
def get_llm_factory() -> ChatModelFactory:
    return ChatModelFactory()
