"""gg.deals client wiring (PR 14)."""

from .client import GgDealsClient

_client: GgDealsClient | None = None


def get_client() -> GgDealsClient:
    """Process-wide client singleton (shares the ICache backend)."""
    global _client
    if _client is None:
        _client = GgDealsClient()
    return _client


def reset_client() -> None:
    """Test hook."""
    global _client
    _client = None
