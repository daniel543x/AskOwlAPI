import logging

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from ...tools.settings import settings
from .selection import ResolvedModel

logger = logging.getLogger(__name__)


class ChatModelFactory:
    """Builds chat models from registry-resolved configuration."""

    def create_chat_model(
        self, resolved: ResolvedModel, reasoning_params: dict | None = None
    ) -> BaseChatModel:
        # Merge order (decision 11): catalog default_params < adapter output
        # (the request wins).
        params = dict(resolved.default_params)
        params.update(reasoning_params or {})

        if resolved.base_url:
            params["base_url"] = resolved.base_url
        if resolved.api_key:
            params["api_key"] = resolved.api_key

        provider = resolved.provider_type.lower()
        if provider == "zai":
            # z.ai serves an OpenAI-compatible endpoint; the reasoning
            # adapter already translated the controls into extra_body.
            model_provider = "openai"
        else:
            model_provider = resolved.provider_type

        if provider == "ollama" and "num_ctx" not in params:
            # Ollama's own default context window silently truncates large
            # prompts (e.g. research synthesis), which makes the model drop
            # the tool instruction and answer in prose. Admins can override
            # per model via the catalog default_params.
            params["num_ctx"] = settings.OLLAMA_NUM_CTX
            logger.debug(
                "Applied default num_ctx=%s to Ollama model '%s'.",
                settings.OLLAMA_NUM_CTX,
                resolved.model_name,
            )

        try:
            return init_chat_model(
                model=resolved.model_name,
                model_provider=model_provider,
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
