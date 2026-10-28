"""Offline unit tests for API-key helpers (no DB)."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException

from src.modules.api_keys.model import (
    ApiKey,
    generate_api_key,
    hash_api_key,
    is_expired,
    validate_scopes,
)
from src.modules.api_keys.rate_limit import RateLimiter
from src.tools.cache import MemoryCache


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


def _key(expires_at: datetime | None) -> ApiKey:
    return ApiKey(
        user_id=uuid.uuid4(),
        name="test",
        key_hash="hash",
        key_prefix="ak-",
        expires_at=expires_at,
    )


def test_is_expired_handles_naive_and_aware_datetimes():
    assert not is_expired(_key(expires_at=None))

    soon = datetime.now(UTC) + timedelta(minutes=5)
    assert not is_expired(_key(expires_at=soon))

    past = datetime.now(UTC) - timedelta(minutes=1)
    assert is_expired(_key(expires_at=past))

    naive_past = (datetime.now(UTC) - timedelta(minutes=1)).replace(tzinfo=None)
    assert is_expired(_key(expires_at=naive_past))


async def test_rate_limiter_fixed_window():
    limiter = RateLimiter(limit_per_minute=2, cache=MemoryCache())
    await limiter.check("k")
    await limiter.check("k")
    with pytest.raises(HTTPException) as exc_info:
        await limiter.check("k")
    assert exc_info.value.status_code == 429
    # The client learns when the window resets.
    assert exc_info.value.headers is not None
    assert int(exc_info.value.headers["Retry-After"]) > 0
    # Other keys unaffected.
    await limiter.check("other")


async def test_rate_limiter_window_reset(monkeypatch):
    import time

    limiter = RateLimiter(limit_per_minute=1, cache=MemoryCache())
    await limiter.check("k")
    with pytest.raises(HTTPException):
        await limiter.check("k")

    # Jump past the window: allowed again.
    real_monotonic = time.monotonic
    monkeypatch.setattr(
        "src.tools.cache.time.monotonic",
        lambda: real_monotonic() + 61,
    )
    await limiter.check("k")
