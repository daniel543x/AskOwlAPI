# user/router.py
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from ...tools.db import get_session  # Upewnij się, że ścieżka jest poprawna
from .model import UserCreate, UserRead, UserUpdate
from .service import create_user, delete_user, get_user_by_id, update_user

router = APIRouter(prefix="/user", tags=["Users"])


@router.post("/", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def create_user_endpoint(
    user_create: UserCreate, session: Session = Depends(get_session)
):
    try:
        db_user = create_user(session=session, user_create=user_create)
        return db_user
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{user_id}", response_model=UserRead)
def read_user(user_id: UUID, session: Session = Depends(get_session)):
    db_user = get_user_by_id(session=session, user_id=user_id)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found."
        )
    return db_user


@router.patch("/{user_id}", response_model=UserRead)
def update_user_endpoint(
    user_id: UUID, user_update: UserUpdate, session: Session = Depends(get_session)
):
    db_user = update_user(session=session, user_id=user_id, user_update=user_update)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found."
        )
    return db_user


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user_endpoint(user_id: UUID, session: Session = Depends(get_session)):
    success = delete_user(session=session, user_id=user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found."
        )
    return None
