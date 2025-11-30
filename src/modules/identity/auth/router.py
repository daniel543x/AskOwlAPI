from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from ....tools.db import get_session
from ....tools.settings import settings
from ..user.model import User, UserRead
from .model import LoginRequest, TokenData
from .security import get_current_user
from .service import authenticate_user, create_access_token

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenData)
def login_for_access_token(
    form_data: LoginRequest,
    session: Session = Depends(get_session),
):
    user = authenticate_user(session, form_data.username_or_email, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    from ..role.service import get_user_roles

    user_roles = get_user_roles(session=session, user_id=user.id)

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email, "roles": user_roles},
        expires_delta=access_token_expires,
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/users/me", response_model=UserRead)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user
