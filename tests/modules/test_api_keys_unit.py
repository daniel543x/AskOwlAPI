"""Offline unit tests for API-key helpers (no DB)."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.modules.api_keys.model import (
    generate_api_key,
    hash_api_key,
    is_expired,
    validate_scopes,
)
from src.modules.api_keys.rate_limit import RateLimiter


def test_generate_api_key_shape():
    raw, key_hash, prefix = generate_api_key()

    assert raw.startswith("ak-")
    assert len(raw) > 40
    assert hash_api_key(raw) == key_hash
    assert prefix.startswith("ak-")
    # Two draws never collide.
    assert generate_api_key()[0] != raw


def test_hash_is_not_reversible_encoding_of_raw():
    raw, key_hash, _ = generate_api_key()
    assert raw not in key_hash


def test_validate_scopes_dedupes_and_rejects_unknown():
    assert validate_scopes(["search", "search", "research"]) == [
        "search",
        "research",
    ]
    with pytest.raises(ValueError, match="root"):
        validate_scopes(["search", "root"])


def test_is_expired_handles_naive_and_aware_datetimes():
    key = SimpleNamespace(expires_at=None)
    assert not is_expired(key)

    soon = datetime.now(UTC) + timedelta(minutes=5)
    assert not is_expired(SimpleNamespace(expires_at=soon))

    past = datetime.now(UTC) - timedelta(minutes=1)
    assert is_expired(SimpleNamespace(expires_at=past))

    naive_past = (datetime.now(UTC) - timedelta(minutes=1)).replace(tzinfo=None)
    assert is_expired(SimpleNamespace(expires_at=naive_past))


def test_rate_limiter_fixed_window():
    limiter = RateLimiter(limit_per_minute=2)
    limiter.check("k")
    limiter.check("k")
    with pytest.raises(HTTPException) as exc_info:
        limiter.check("k")
    assert exc_info.value.status_code == 429
    # Other keys unaffected.
    limiter.check("other")


def test_rate_limiter_window_reset(monkeypatch):
    import time

    limiter = RateLimiter(limit_per_minute=1)
    limiter.check("k")
    with pytest.raises(HTTPException):
        limiter.check("k")

    # Jump past the window: allowed again.
    real_monotonic = time.monotonic
    monkeypatch.setattr(
        "src.modules.api_keys.rate_limit.time.monotonic",
        lambda: real_monotonic() + 61,
    )
    limiter.check("k")
