from datetime import datetime, timedelta
from typing import Optional, cast

from jose import JWTError, jwt
from sqlmodel import Session, select

from ...tools.settings import settings
from ..user.model import User
from .model import JWTPayload
from .passwd import check_passwd


def authenticate_user(session: Session, email: str, password: str) -> User | None:
    user = session.exec(select(User).where(User.email == email)).first()
    if not user:
        return None
    if check_passwd(password, user.password_hash) is False:
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
