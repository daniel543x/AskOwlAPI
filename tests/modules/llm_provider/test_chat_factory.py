"""ChatModelFactory: Ollama context-window defaults (offline tests)."""

from src.modules.llm_provider.factory import ChatModelFactory
from src.modules.llm_provider.selection import ResolvedModel
from src.tools.settings import settings


def _resolved(
    provider_type: str, default_params: dict[str, object] | None = None
) -> ResolvedModel:
    return ResolvedModel(
        kind="chat",
        model_name="some-model",
        provider_type=provider_type,
        base_url="http://localhost:11434",
        api_key="test-key",
        default_params=default_params or {},
    )


def test_ollama_models_get_the_default_context_window():
    model = ChatModelFactory().create_chat_model(_resolved("ollama"))

    assert getattr(model, "num_ctx", None) == settings.OLLAMA_NUM_CTX


def test_catalog_num_ctx_overrides_the_default():
    model = ChatModelFactory().create_chat_model(
        _resolved("ollama", {"num_ctx": 8192})
    )

    assert getattr(model, "num_ctx", None) == 8192


def test_non_ollama_models_do_not_get_num_ctx():
    model = ChatModelFactory().create_chat_model(_resolved("openai"))

    assert not hasattr(model, "num_ctx")
