"""Offline unit tests for refresh-token helpers (no DB)."""

import hashlib
import uuid
from datetime import UTC, datetime, timedelta

from src.modules.identity.auth.model import (
    REFRESH_TOKEN_PREFIX,
    RefreshToken,
    chain_descendants,
    generate_refresh_token,
    refresh_expiry,
    token_state,
)
from src.tools.settings import settings


def _row(
    parent_id: uuid.UUID | None = None,
    *,
    revoked_at: datetime | None = None,
    rotated_at: datetime | None = None,
    expires_in: timedelta = timedelta(days=90),
) -> RefreshToken:
    return RefreshToken(
        user_id=uuid.uuid4(),
        token_hash=uuid.uuid4().hex,
        parent_id=parent_id,
        revoked_at=revoked_at,
        rotated_at=rotated_at,
        expires_at=datetime.now(UTC) + expires_in,
    )


def test_generate_refresh_token_shape_and_uniqueness():
    raw, token_hash = generate_refresh_token()

    assert raw.startswith(REFRESH_TOKEN_PREFIX)
    assert len(raw) > 40
    assert token_hash == hashlib.sha256(raw.encode()).hexdigest()
    assert generate_refresh_token()[0] != raw


def test_hash_is_not_reversible_encoding_of_raw():
    raw, token_hash = generate_refresh_token()
    assert raw not in token_hash


def test_refresh_expiry_uses_configured_window():
    now = datetime.now(UTC)
    expires = refresh_expiry(now)
    assert (expires - now).days == settings.REFRESH_TOKEN_EXPIRE_DAYS


def test_token_state_classification():
    now = datetime.now(UTC)
    assert token_state(_row(), now) == "active"
    assert token_state(_row(expires_in=timedelta(days=-1)), now) == "expired"
    assert token_state(_row(rotated_at=now), now) == "rotated"
    assert token_state(_row(revoked_at=now), now) == "revoked"


def test_rotated_wins_over_expired_for_reuse_detection():
    now = datetime.now(UTC)
    token = _row(
        rotated_at=now - timedelta(days=91), expires_in=timedelta(days=-1)
    )
    # Replaying a rotated token must be reuse detection, never a benign
    # "expired" answer that would leave the chain alive.
    assert token_state(token, now) == "rotated"


def test_naive_expiry_is_treated_as_utc():
    token = _row(expires_in=timedelta(days=-1))
    token.expires_at = token.expires_at.replace(tzinfo=None)
    assert token_state(token) == "expired"


def test_chain_descendants_walks_the_rotation_chain():
    root = _row()
    child = _row(parent_id=root.id)
    grandchild = _row(parent_id=child.id)
    unrelated = _row()

    chain = chain_descendants([root, child, grandchild, unrelated], root)

    assert sorted(row.id for row in chain) == sorted(
        [root.id, child.id, grandchild.id]
    )


def test_chain_descendants_survives_a_cycle():
    root = _row()
    child = _row(parent_id=root.id)
    root.parent_id = child.id

    chain = chain_descendants([root, child], root)

    assert sorted(row.id for row in chain) == sorted([root.id, child.id])
