import pytest

import src.modules.owl_reranker.factory as ranker_factory
from src.tools.settings import settings


class FakeOwlRanker:
    def __init__(self, model_name=None, cache_dir=None):
        self.model_name = model_name
        self.cache_dir = cache_dir


async def test_get_ranker_fails_before_initialization(monkeypatch):
    monkeypatch.setattr(ranker_factory, "_ranker", None)

    with pytest.raises(RuntimeError, match="lifespan"):
        ranker_factory.get_ranker()


async def test_init_ranker_loads_model_once_with_settings(monkeypatch):
    monkeypatch.setattr(ranker_factory, "_ranker", None)
    created = []
    monkeypatch.setattr(
        ranker_factory,
        "OwlRanker",
        lambda model_name, cache_dir: created.append(model_name) or FakeOwlRanker(),
    )

    first = await ranker_factory.init_ranker()
    second = await ranker_factory.init_ranker()

    assert first is second
    assert isinstance(ranker_factory.get_ranker(), FakeOwlRanker)
    assert created == [settings.RANKER_MODEL_NAME]
