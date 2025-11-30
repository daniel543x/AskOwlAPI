from typing import Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from ..auth.passwd import hash_passwd
from .model import User, UserCreate, UserUpdate


async def get_user_by_email(session: AsyncSession, email: str) -> Optional[User]:
    statement = select(User).where(User.email == email)
    result = await session.execute(statement)
    return result.scalar_one_or_none()


async def get_user_by_nickname(session: AsyncSession, nickname: str) -> Optional[User]:
    statement = select(User).where(User.nickname == nickname)
    result = await session.execute(statement)
    return result.scalar_one_or_none()


async def get_user_by_id(session: AsyncSession, user_id: UUID) -> Optional[User]:
    return await session.get(User, user_id)


async def get_user_by_identifier(
    session: AsyncSession, identifier: str
) -> Optional[User]:
    user = await get_user_by_email(session, identifier)
    if user:
        return user
    user = await get_user_by_nickname(session, identifier)
    return user


async def create_user(session: AsyncSession, user_create: UserCreate) -> User:
    existing_user = await get_user_by_email(session, user_create.email)
    if existing_user:
        raise ValueError("User with this email already exists")

    existing_user = await get_user_by_nickname(session, user_create.nickname)
    if existing_user:
        raise ValueError("User with this nickname already exists")

    db_user = User(
        email=user_create.email,
        nickname=user_create.nickname,
        password_hash=await hash_passwd(user_create.password),
    )

    session.add(db_user)
    await session.commit()
    await session.refresh(db_user)
    return db_user


async def update_user(
    session: AsyncSession, user_id: UUID, user_update: UserUpdate
) -> Optional[User]:
    db_user = await session.get(User, user_id)
    if not db_user:
        return None

    if user_update.password:
        db_user.password_hash = await hash_passwd(user_update.password)

    if user_update.email and user_update.email != db_user.email:
        existing_user = await get_user_by_email(session, user_update.email)
        if existing_user:
            raise ValueError("User with this email already exists")
        db_user.email = user_update.email

    if user_update.nickname and user_update.nickname != db_user.nickname:
        existing_user = await get_user_by_nickname(session, user_update.nickname)
        if existing_user:
            raise ValueError("User with this nickname already exists")
        db_user.nickname = user_update.nickname

    session.add(db_user)
    await session.commit()
    await session.refresh(db_user)
    return db_user


async def delete_user(session: AsyncSession, user_id: UUID) -> bool:
    user = await session.get(User, user_id)
    if not user:
        return False
    await session.delete(user)
    await session.commit()
    return True
