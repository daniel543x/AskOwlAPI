import uuid
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import (
    APIKeyHeader,
    HTTPAuthorizationCredentials,
    HTTPBearer,
    # OAuth2PasswordBearer,
)
from sqlmodel.ext.asyncio.session import AsyncSession

from ....tools.db import get_session
from ..user.model import User
from .service import get_user_by_api_key, verify_token

# https://fastapi.tiangolo.com/reference/openapi/models/
# SecurityScheme = (
#    APIKey | HTTPBase | OAuth2 | OpenIdConnect | HTTPBearer
# )
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# https://fastapi.tiangolo.com/reference/openapi/models/#fastapi.openapi.models.HTTPBase.model_config
bearer_scheme = HTTPBearer(auto_error=False)
# https://fastapi.tiangolo.com/reference/openapi/models/#fastapi.openapi.models.APIKey
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user(
    # token: str = Depends(oauth2_scheme),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    key: Optional[str] = Depends(api_key_scheme),
    session: AsyncSession = Depends(get_session),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials. Use Bearer token or X-API-Key.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    # <----- Chcek credentials ----->
    if credentials:
        token = credentials.credentials
        try:
            payload = verify_token(token)
            if payload:
                user_id_str = payload.get("sub")
                user_id = uuid.UUID(user_id_str)
                user = await session.get(User, user_id)
                if user:
                    return user
        except Exception:
            raise credentials_exception

    # <----- Chcek credentials ----->
    if key:
        try:
            user = await get_user_by_api_key(session, key)
            if user:
                return user
        except Exception:
            raise credentials_exception

    raise credentials_exception


async def get_current_admin_user(
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> User:
    from ..role.service import (
        get_user_roles,
    )

    user_roles = await get_user_roles(session=session, user_id=current_user.id)

    if "admin" not in user_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )
    return current_user
