from src.modules.search_engine.caching import CachedSearchProvider
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


# <----- Factory-level cache wiring (WU-10) ----->


def test_factory_reuses_one_cached_wrapper_per_config(monkeypatch):
    """The factory runs per request; the cache must outlive the request."""
    from src.modules.search_engine.factory import SearchProviderFactory
    from src.tools.settings import settings

    monkeypatch.setattr(settings, "SEARCH_CACHE", True)
    SearchProviderFactory._cache_wrappers.clear()
    try:
        first = SearchProviderFactory._provider_for("searxng", {})
        second = SearchProviderFactory._provider_for("searxng", {})
        other_config = SearchProviderFactory._provider_for(
            "searxng", {"engines": "duckduckgo"}
        )

        assert isinstance(first, CachedSearchProvider)
        assert isinstance(other_config, CachedSearchProvider)
        assert first is second, "same config must reuse the same wrapper"
        assert first is not other_config, "different config gets its own wrapper"
        # The inner provider is shared too, so memoization actually persists.
        assert other_config.inner is not first.inner
    finally:
        SearchProviderFactory._cache_wrappers.clear()


def test_factory_cache_off_returns_raw_provider(monkeypatch):
    from src.modules.search_engine.factory import SearchProviderFactory
    from src.tools.settings import settings

    monkeypatch.setattr(settings, "SEARCH_CACHE", False)

    provider = SearchProviderFactory._provider_for("searxng", {})

    assert isinstance(provider, SearXNGProvider)
    assert not isinstance(provider, CachedSearchProvider)
