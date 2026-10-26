from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from ....tools.db import get_session
from ..auth.security import get_current_admin_user
from .model import User, UserAdminUpdate, UserCreate, UserRead, UserUpdate
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e


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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e


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


# <----- Admin: user management ----- #

admin_router = APIRouter(
    prefix="/admin/users",
    tags=["Admin Users"],
    dependencies=[Depends(get_current_admin_user)],
)


@admin_router.get("", response_model=list[UserRead])
async def admin_list_users(
    session: AsyncSession = Depends(get_session),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    statement = select(User).order_by(col(User.created_at)).offset(offset).limit(limit)
    result = await session.execute(statement)
    return list(result.scalars().all())


@admin_router.patch("/{user_id}/status", response_model=UserRead)
async def admin_set_user_status(
    user_id: UUID,
    changes: UserAdminUpdate,
    session: AsyncSession = Depends(get_session),
    admin: User = Depends(get_current_admin_user),
):
    """Block or unblock a user; blocked users fail auth everywhere."""
    if changes.is_active is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Nothing to update: provide is_active.",
        )
    if user_id == admin.id and changes.is_active is False:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Admins cannot block their own account.",
        )

    user = await get_user_by_id(session=session, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found."
        )

    user.is_active = changes.is_active
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user
