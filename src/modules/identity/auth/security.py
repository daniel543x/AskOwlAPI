import uuid

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import (
    APIKeyHeader,
    HTTPAuthorizationCredentials,
    HTTPBearer,
)
from sqlmodel.ext.asyncio.session import AsyncSession

from ....tools.cache import build_cache
from ....tools.db import get_session
from ....tools.settings import settings
from ...api_keys.rate_limit import RateLimiter
from ..user.model import User
from .service import verify_token

# https://fastapi.tiangolo.com/reference/openapi/models/
bearer_scheme = HTTPBearer(auto_error=False)
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)

_rate_limiter = RateLimiter(
    settings.API_KEY_RATE_LIMIT_PER_MINUTE, cache=build_cache()
)

credentials_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials. Use Bearer token or X-API-Key.",
    headers={"WWW-Authenticate": "Bearer"},
)


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    key: str | None = Depends(api_key_scheme),
    session: AsyncSession = Depends(get_session),
) -> User:
    # <----- Bearer token (JWT) ----->
    if credentials:
        try:
            payload = verify_token(credentials.credentials)
            if payload:
                user_id = uuid.UUID(payload.get("sub"))
                user = await session.get(User, user_id)
                if user:
                    if not user.is_active:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="User account is disabled.",
                        )
                    await session.refresh(user, attribute_names=["roles"])
                    # JWT sessions are not scope-limited (role gates apply).
                    request.state.api_key_scopes = None
                    return user
        except HTTPException:
            raise
        except Exception as exc:
            raise credentials_exception from exc

    # <----- API key (X-API-Key) ----->
    if key:
        from ...api_keys.repository import ApiKeyRepository

        resolved = await ApiKeyRepository(session).resolve(key)
        if resolved is None:
            raise credentials_exception
        api_key, user = resolved

        await _rate_limiter.check(api_key.id)
        request.state.api_key_scopes = list(api_key.scopes)
        return user

    raise credentials_exception


def require_scope(scope: str):
    """Mode gate for API-key sessions: the key must carry `scope`.

    JWT sessions pass through (role gating already applies via the
    elevated-user dependencies); this only narrows API keys.
    """

    async def dependency(
        request: Request,
        user: User = Depends(get_current_user),
    ) -> User:
        scopes = getattr(request.state, "api_key_scopes", None)
        if scopes is not None and scope not in scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This mode requires the '{scope}' scope on the API key.",
            )
        return user

    return dependency


async def get_current_admin_user(
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> User:
    from ..role.service import get_user_roles

    user_roles = await get_user_roles(session=session, user_id=current_user.id)

    if "admin" not in user_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )
    return current_user


# Roles allowed to use the elevated modes (`research`, `agent`).
ELEVATED_ROLES = ("research", "admin")


async def get_current_elevated_user(
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> User:
    from ..role.service import get_user_roles

    user_roles = await get_user_roles(session=session, user_id=current_user.id)

    if not any(role in ELEVATED_ROLES for role in user_roles):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "This mode requires an elevated role "
                f"({' or '.join(ELEVATED_ROLES)})."
            ),
        )
    return current_user
