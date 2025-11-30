from datetime import datetime, timedelta
from typing import Optional, cast

from jose import JWTError, jwt
from sqlmodel.ext.asyncio.session import AsyncSession

from ....tools.settings import settings
from ..user.model import User
from ..user.service import get_user_by_identifier
from .model import JWTPayload
from .passwd import check_passwd


async def authenticate_user(
    session: AsyncSession, identifier: str, password: str
) -> Optional[User]:
    user = await get_user_by_identifier(session, identifier)
    if not user:
        return None
    if await check_passwd(password, user.password_hash) is False:
        return None
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )
    return encoded_jwt


def verify_token(token: str) -> Optional[JWTPayload]:
    try:
        payload = cast(
            JWTPayload,
            jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]),
        )
        email: str = payload.get("sub")
        if email is None:
            return None
        return payload
    except JWTError:
        return None
