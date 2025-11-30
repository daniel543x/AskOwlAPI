from typing import Optional
from uuid import UUID

from sqlmodel import Session, select

from ..auth.passwd import hash_passwd
from .model import User, UserCreate, UserUpdate


def get_user_by_email(session: Session, email: str) -> Optional[User]:
    statement = select(User).where(User.email == email)
    user = session.exec(statement).first()
    return user


def get_user_by_nickname(session: Session, nickname: str) -> Optional[User]:
    statement = select(User).where(User.nickname == nickname)
    user = session.exec(statement).first()
    return user


def get_user_by_id(session: Session, user_id: UUID) -> Optional[User]:
    return session.get(User, user_id)


def get_user_by_identifier(session: Session, identifier: str) -> Optional[User]:
    user = get_user_by_email(session, identifier)
    if user:
        return user
    user = get_user_by_nickname(session, identifier)
    return user


def create_user(session: Session, user_create: UserCreate) -> User:
    existing_user = get_user_by_email(session, user_create.email)
    if existing_user:
        raise ValueError("User with this email already exists")

    existing_user = get_user_by_nickname(session, user_create.nickname)
    if existing_user:
        raise ValueError("User with this nickname already exists")

    db_user = User(
        email=user_create.email,
        nickname=user_create.nickname,
        password_hash=hash_passwd(user_create.password),
    )

    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user


def update_user(
    session: Session, user_id: UUID, user_update: UserUpdate
) -> Optional[User]:
    db_user = session.get(User, user_id)
    if not db_user:
        return None

    if user_update.password:
        db_user.password_hash = hash_passwd(user_update.password)

    if user_update.email:
        db_user.email = user_update.email

    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user


def delete_user(session: Session, user_id: UUID) -> bool:
    user = session.get(User, user_id)
    if not user:
        return False
    session.delete(user)
    session.commit()
    return True
