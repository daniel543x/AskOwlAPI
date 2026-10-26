from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlmodel.ext.asyncio.session import AsyncSession

from ....tools.db import get_session
from ..user.model import User, UserRead
from .model import (
    CLIENT_LABEL_MAX_LENGTH,
    LoginRequest,
    LogoutRequest,
    RefreshRequest,
    TokenData,
)
from .security import get_current_user
from .service import (
    InvalidRefreshToken,
    login_session,
    logout_session,
    refresh_session,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


def _client_label(request: Request) -> str:
    """Non-secret session hint from the User-Agent header."""
    return request.headers.get("user-agent", "").strip()[:CLIENT_LABEL_MAX_LENGTH]


@router.post("/login", response_model=TokenData)
async def login_for_access_token(
    form_data: LoginRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Authenticate and issue an access token plus a refresh token.

    The refresh token (shown once, hash-only server-side) renews the
    session via `POST /auth/refresh` without re-sending the password.
    """
    tokens = await login_session(
        session,
        form_data.username_or_email,
        form_data.password,
        client_label=_client_label(request),
    )
    if tokens is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    _, access_token, refresh_token = tokens
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "refresh_token": refresh_token,
    }


@router.post("/refresh", response_model=TokenData)
async def refresh_access_token(
    body: RefreshRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Single-use rotation: a refresh token is valid exactly once.

    Presenting an already-rotated or revoked token revokes the whole
    rotation chain (stolen-token detection) and returns 401.
    """
    try:
        _, access_token, refresh_token = await refresh_session(
            session, body.refresh_token, client_label=_client_label(request)
        )
    except InvalidRefreshToken as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "refresh_token": refresh_token,
    }


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    body: LogoutRequest,
    session: AsyncSession = Depends(get_session),
):
    """Revoke the refresh-token session; idempotent (always 204)."""
    await logout_session(session, body.refresh_token)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/users/me", response_model=UserRead)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user
