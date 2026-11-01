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


# <----- Catalog provenance on the built model (webapp API-NOTES #11) ----->


def test_chat_models_carry_catalog_metadata():
    import uuid

    resolved = _resolved("ollama")
    resolved.entry_id = uuid.uuid4()
    resolved.display_name = "Qwen 3.5"
    resolved.provider_name = "Local Ollama"

    model = ChatModelFactory().create_chat_model(resolved)

    assert model.metadata is not None
    assert model.metadata["catalog_entry_id"] == str(resolved.entry_id)
    assert model.metadata["catalog_label"] == "Qwen 3.5"
    assert model.metadata["catalog_provider"] == "Local Ollama"


def test_builtin_models_fall_back_to_the_model_name_label():
    resolved = _resolved("ollama")

    model = ChatModelFactory().create_chat_model(resolved)

    assert model.metadata is not None
    assert model.metadata["catalog_entry_id"] is None
    assert model.metadata["catalog_label"] == resolved.model_name
    assert model.metadata["catalog_provider"] == "ollama"
