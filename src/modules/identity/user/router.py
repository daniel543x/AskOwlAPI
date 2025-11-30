from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ....tools.db import get_session
from .model import UserCreate, UserRead, UserUpdate
from .service import (
    create_user,
    delete_user,
    get_user_by_id,
    get_user_by_identifier,
    update_user,
)

router = APIRouter(prefix="/user", tags=["Users"])


@router.post("/", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def create_user_endpoint(
    user_create: UserCreate, session: AsyncSession = Depends(get_session)
):
    try:
        db_user = await create_user(session=session, user_create=user_create)
        return db_user
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{user_id}", response_model=UserRead)
async def read_user(user_id: UUID, session: AsyncSession = Depends(get_session)):
    db_user = await get_user_by_id(session=session, user_id=user_id)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found."
        )
    return db_user


@router.get("/nn/{nickname}", response_model=UserRead)
async def read_user_by_nickname(
    nickname: str, session: AsyncSession = Depends(get_session)
):
    db_user = await get_user_by_identifier(session=session, identifier=nickname)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found."
        )
    return db_user


@router.patch("/{user_id}", response_model=UserRead)
async def update_user_endpoint(
    user_id: UUID, user_update: UserUpdate, session: AsyncSession = Depends(get_session)
):
    try:
        db_user = await update_user(
            session=session, user_id=user_id, user_update=user_update
        )
        if not db_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found."
            )
        return db_user
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_endpoint(
    user_id: UUID, session: AsyncSession = Depends(get_session)
):
    success = await delete_user(session=session, user_id=user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found."
        )
    return
