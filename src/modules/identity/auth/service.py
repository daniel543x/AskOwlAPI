import logging
from datetime import UTC, datetime, timedelta
from typing import cast

import jwt
from jwt.exceptions import InvalidTokenError
from sqlmodel.ext.asyncio.session import AsyncSession

from ....tools.settings import settings
from ..user.model import User
from ..user.service import get_user_by_identifier
from .model import JWTPayload, token_state
from .passwd import check_passwd
from .repository import RefreshTokenRepository

logger = logging.getLogger(__name__)


class InvalidRefreshToken(Exception):
    """Unknown, expired, rotated, or revoked refresh token."""


async def authenticate_user(
    session: AsyncSession, identifier: str, password: str
) -> User | None:
    user = await get_user_by_identifier(session, identifier)
    if not user:
        return None
    if not user.is_active:
        # Disabled accounts fail login exactly like unknown credentials.
        return None
    if await check_passwd(password, user.password_hash) is False:
        return None
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.MASTER_KEY, algorithm=settings.ALGORITHM
    )
    return encoded_jwt


def verify_token(token: str) -> JWTPayload | None:
    try:
        payload = cast(
            JWTPayload,
            jwt.decode(token, settings.MASTER_KEY, algorithms=[settings.ALGORITHM]),
        )
        subject: str = payload.get("sub")
        if subject is None:
            return None
        return payload
    except InvalidTokenError:
        return None


async def get_user_by_api_key(session: AsyncSession, key: str):
    """Deprecated stub kept for import compatibility; resolution lives in
    ApiKeyRepository.resolve (hash lookup, activity/expiry checks)."""
    from ...api_keys.repository import ApiKeyRepository

    resolved = await ApiKeyRepository(session).resolve(key)
    return resolved[1] if resolved else None


async def _access_token_for(session: AsyncSession, user: User) -> str:
    from ..role.service import get_user_roles

    user_roles = await get_user_roles(session=session, user_id=user.id)
    return create_access_token(
        data={"sub": str(user.id), "roles": user_roles},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )


async def login_session(
    session: AsyncSession, identifier: str, password: str, client_label: str
) -> tuple[User, str, str] | None:
    """Authenticate and mint the access + refresh pair (None on bad login)."""
    user = await authenticate_user(session, identifier, password)
    if user is None:
        return None

    access_token = await _access_token_for(session, user)
    _, refresh_raw = await RefreshTokenRepository(session).create(
        user.id, client_label=client_label
    )
    return user, access_token, refresh_raw


async def refresh_session(
    session: AsyncSession, raw_refresh_token: str, client_label: str
) -> tuple[User, str, str]:
    """Rotate a refresh token into a fresh access + refresh pair.

    Single use: the presented row is marked rotated and a child row carries
    the session forward. Replaying a rotated or revoked token is treated as
    theft — the whole rotation chain is revoked before failing.
    """
    repository = RefreshTokenRepository(session)
    token = await repository.get_by_raw(raw_refresh_token)
    if token is None:
        raise InvalidRefreshToken()

    state = token_state(token)
    if state in ("rotated", "revoked"):
        revoked = await repository.revoke_chain(token)
        logger.warning(
            "Refresh-token reuse detected (state=%s); revoked %d token(s) "
            "in the rotation chain of user %s.",
            state,
            revoked,
            token.user_id,
        )
        raise InvalidRefreshToken()
    if state == "expired":
        raise InvalidRefreshToken()

    user = await session.get(User, token.user_id)
    if user is None or not user.is_active:
        await repository.revoke_chain(token)
        raise InvalidRefreshToken()

    _, refresh_raw = await repository.rotate(token, client_label=client_label)
    access_token = await _access_token_for(session, user)
    return user, access_token, refresh_raw


async def logout_session(session: AsyncSession, raw_refresh_token: str) -> None:
    """Revoke the presented session chain; silent on unknown tokens."""
    repository = RefreshTokenRepository(session)
    token = await repository.get_by_raw(raw_refresh_token)
    if token is None:
        return
    await repository.revoke_chain(token)
