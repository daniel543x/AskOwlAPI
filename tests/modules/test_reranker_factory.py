import pytest

from src.modules.llm_provider.selection import ResolvedModel
from src.modules.owl_reranker.factory import RankerFactory
from src.tools.settings import settings


class FakeOwlRanker:
    def __init__(self, model_name=None, cache_dir=None):
        self.model_name = model_name
        self.cache_dir = cache_dir


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(RankerFactory, "REGISTRY", {"flashrank": FakeOwlRanker})
    monkeypatch.setattr(RankerFactory, "_instances", {})
    yield


async def test_get_or_load_caches_per_model():
    first = await RankerFactory.get_or_load("flashrank", "model-a", "/tmp/x")
    second = await RankerFactory.get_or_load("flashrank", "model-a", "/tmp/x")
    other = await RankerFactory.get_or_load("flashrank", "model-b", "/tmp/x")

    assert first is second
    assert first is not other
    assert first.model_name == "model-a"
    assert other.model_name == "model-b"


async def test_get_or_load_unknown_provider_type():
    with pytest.raises(ValueError, match="No ranker implementation"):
        await RankerFactory.get_or_load("mystery", "model-a")


async def test_from_resolved_maps_cache_dir():
    resolved = ResolvedModel(
        kind="reranker",
        model_name="ms-marco-MiniLM-L-12-v2",
        provider_type="flashrank",
        default_params={"cache_dir": "/tmp/custom"},
    )

    ranker = await RankerFactory.from_resolved(resolved)
    assert ranker.cache_dir == "/tmp/custom"
    assert ranker.model_name == "ms-marco-MiniLM-L-12-v2"


async def test_get_or_load_defaults_cache_dir_to_none():
    ranker = await RankerFactory.get_or_load("flashrank", "model-a")
    assert ranker.cache_dir is None


def test_reset_cache_drops_instances():
    RankerFactory._instances[("flashrank", "m", "")] = FakeOwlRanker()
    RankerFactory.reset_cache()
    assert RankerFactory._instances == {}


async def test_builtin_resolved_model_uses_settings():
    # The built-in tier (fresh DB) resolves to the settings-configured model.
    resolved = ResolvedModel(
        kind="reranker",
        model_name=settings.RANKER_MODEL_NAME,
        provider_type="flashrank",
        default_params={"cache_dir": settings.RANKER_CACHE_DIR},
    )
    ranker = await RankerFactory.from_resolved(resolved)
    assert ranker.model_name == settings.RANKER_MODEL_NAME
