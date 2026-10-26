"""Refresh-token session flow (integration: live Postgres)."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from httpx import Response

from src.modules.identity.auth.repository import RefreshTokenRepository
from src.modules.identity.role.service import assign_role_to_user
from src.modules.identity.user.model import UserCreate
from src.modules.identity.user.service import create_user
from src.tools.settings import settings

pytestmark = pytest.mark.integration

PASSWORD = "Sup3rSecret!pass"


async def make_user(db_session, role_name: str | None = None):
    suffix = uuid.uuid4().hex[:8]
    user = await create_user(
        db_session,
        UserCreate(
            email=f"{suffix}@example.com",
            nickname=f"user_{suffix}",
            password=PASSWORD,
        ),
    )
    if role_name:
        assert await assign_role_to_user(db_session, user.id, role_name)
    return user


async def login(client, user) -> dict:
    response = await client.post(
        "/auth/login",
        json={"username_or_email": user.email, "password": PASSWORD},
    )
    assert response.status_code == 200, response.text
    return response.json()


async def refresh(client, refresh_token: str) -> Response:
    return await client.post("/auth/refresh", json={"refresh_token": refresh_token})


# <----- Login issues a session -----


async def test_login_returns_refresh_token_and_stores_only_hash(client, db_session):
    user = await make_user(db_session)

    tokens = await login(client, user)

    assert tokens["token_type"] == "bearer"
    assert tokens["refresh_token"].startswith("rt-")
    row = await RefreshTokenRepository(db_session).get_by_raw(
        tokens["refresh_token"]
    )
    assert row is not None
    assert tokens["refresh_token"] not in row.token_hash


# <----- Rotation is single use -----


async def test_refresh_rotates_and_retires_the_old_token(client, db_session):
    user = await make_user(db_session)
    tokens = await login(client, user)

    response = await refresh(client, tokens["refresh_token"])
    assert response.status_code == 200, response.text
    rotated = response.json()
    assert rotated["refresh_token"] != tokens["refresh_token"]
    assert rotated["access_token"]

    replay = await refresh(client, tokens["refresh_token"])
    assert replay.status_code == 401


async def test_replaying_a_rotated_token_kills_the_whole_chain(client, db_session):
    user = await make_user(db_session)
    first = await login(client, user)
    second = (await refresh(client, first["refresh_token"])).json()

    replay = await refresh(client, first["refresh_token"])
    assert replay.status_code == 401

    # The chain revocation also invalidates the token the legitimate
    # client currently holds.
    current = await refresh(client, second["refresh_token"])
    assert current.status_code == 401


async def test_unknown_refresh_token_is_a_generic_401(client, db_session):
    response = await refresh(client, "rt-not-a-real-token")
    assert response.status_code == 401
    assert "Invalid" in response.json()["detail"]


# <----- Logout, expiry, blocking, password change -----


async def test_logout_revokes_the_session_and_is_idempotent(client, db_session):
    user = await make_user(db_session)
    tokens = await login(client, user)

    response = await client.post(
        "/auth/logout", json={"refresh_token": tokens["refresh_token"]}
    )
    assert response.status_code == 204
    assert (await refresh(client, tokens["refresh_token"])).status_code == 401

    # Logging out again with the same token stays silent.
    again = await client.post(
        "/auth/logout", json={"refresh_token": tokens["refresh_token"]}
    )
    assert again.status_code == 204


async def test_expired_token_rejected_but_row_kept_for_retention(client, db_session):
    user = await make_user(db_session)
    repository = RefreshTokenRepository(db_session)
    _, raw = await repository.create(user.id)
    row = await repository.get_by_raw(raw)
    assert row is not None
    row.expires_at = datetime.now(UTC) - timedelta(days=1)
    db_session.add(row)
    await db_session.commit()

    assert (await refresh(client, raw)).status_code == 401
    assert await repository.get_by_raw(raw) is not None


async def test_blocked_user_cannot_refresh(client, db_session):
    user = await make_user(db_session)
    admin = await make_user(db_session, role_name="admin")
    tokens = await login(client, user)

    from src.modules.identity.auth.service import create_access_token

    block = await client.patch(
        f"/admin/users/{user.id}/status",
        json={"is_active": False},
        headers={
            "Authorization": "Bearer "
            + create_access_token({"sub": str(admin.id)})
        },
    )
    assert block.status_code == 200, block.text
    assert (await refresh(client, tokens["refresh_token"])).status_code == 401


async def test_password_change_revokes_live_sessions(client, db_session):
    user = await make_user(db_session)
    tokens = await login(client, user)

    from src.modules.identity.auth.service import create_access_token

    change = await client.patch(
        f"/user/{user.id}",
        json={"password": "An0therSecret!pass"},
        headers={
            "Authorization": "Bearer "
            + create_access_token({"sub": str(user.id)})
        },
    )
    assert change.status_code == 200, change.text
    assert (await refresh(client, tokens["refresh_token"])).status_code == 401


# <----- Retention purge -----


async def test_purge_deletes_aged_rows_and_keeps_live_ones(client, db_session):
    user = await make_user(db_session)
    repository = RefreshTokenRepository(db_session)

    _, aged_raw = await repository.create(user.id)
    row = await repository.get_by_raw(aged_raw)
    assert row is not None
    row.expires_at = datetime.now(UTC) - timedelta(
        days=settings.REFRESH_TOKEN_RETENTION_DAYS + 1
    )
    db_session.add(row)
    await db_session.commit()

    _, live_raw = await repository.create(user.id)

    purged = await repository.purge_expired(settings.REFRESH_TOKEN_RETENTION_DAYS)

    assert purged >= 1
    assert await repository.get_by_raw(aged_raw) is None
    assert await repository.get_by_raw(live_raw) is not None
