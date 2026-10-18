from src.modules.search_engine.factory import filter_provider_kwargs
from src.modules.search_engine.providers.searxng import SearXNGProvider


def test_filter_keeps_only_constructor_kwargs():
    settings_dict = {
        "base_url": "http://searxng:8080",
        "engines": "duckduckgo",
        "unsupported_key": "dropped",
        "another_bogus": 42,
    }

    filtered = filter_provider_kwargs(SearXNGProvider, settings_dict)

    assert filtered == {"base_url": "http://searxng:8080", "engines": "duckduckgo"}


def test_filter_handles_empty_settings():
    assert filter_provider_kwargs(SearXNGProvider, {}) == {}


def test_searxng_provider_accepts_constructor_params():
    provider = SearXNGProvider(
        base_url="http://searxng:8080", engines="duckduckgo", timeout=5.0
    )

    assert provider.base_url == "http://searxng:8080"
    assert provider.engines == "duckduckgo"
    assert provider.timeout == 5.0


def test_searxng_provider_falls_back_to_settings_defaults(monkeypatch):
    from src.tools.settings import settings

    provider = SearXNGProvider()

    assert provider.base_url == settings.SEARXNG_BASE_URL
    assert provider.engines == settings.SEARXNG_ENGINES
    assert provider.categories == settings.SEARXNG_CATEGORIES
    assert provider.timeout == settings.SEARXNG_TIMEOUT
