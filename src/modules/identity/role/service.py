from typing import List
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlmodel import Session, select

from .model import Role, RoleCreate, UserRole


def create_role(session: Session, role: "RoleCreate") -> Role:
    db_role = Role.model_validate(role)
    session.add(db_role)
    session.commit()
    session.refresh(db_role)
    return db_role


def get_role_by_name(session: Session, name: str) -> Role | None:
    statement = select(Role).where(Role.name == name)
    return session.exec(statement).first()


def assign_role_to_user(session: Session, user_id: UUID, role_name: str) -> bool:
    from ..user.model import User

    user = session.get(User, user_id)
    role = get_role_by_name(session, role_name)

    if not user or not role:
        return False

    user_role_link = UserRole(user_id=user_id, role_id=role.id)
    session.add(user_role_link)
    session.commit()
    return True


def remove_role_from_user(session: Session, user_id: UUID, role_name: str) -> None:
    role = get_role_by_name(session, role_name)
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Role '{role_name}' not found.",
        )

    if role.name.lower() == "admin":
        admin_role_id = role.id
        statement = select(func.count()).where(UserRole.role_id == admin_role_id)
        admin_count = session.exec(statement).one()

        if admin_count <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot remove the last administrator role from the system.",
            )

    statement = select(UserRole).where(
        UserRole.user_id == user_id, UserRole.role_id == role.id
    )
    user_role_link = session.exec(statement).first()

    if not user_role_link:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User does not have this role.",
        )

    session.delete(user_role_link)
    session.commit()


def get_user_roles(session: Session, user_id: UUID) -> List[str]:
    statement = select(Role.name).join(UserRole).where(UserRole.user_id == user_id)
    return list(session.exec(statement).all())


# --- Auxiliary function for creating default roles. ---
def init_roles(session: Session) -> None:
    default_roles_data = [
        {
            "name": "user",
            "description": "Default user with standard permissions.",
        },
        {
            "name": "admin",
            "description": "Administrator with full system access.",
        },
    ]

    for role_info in default_roles_data:
        role_name = role_info["name"]

        role = get_role_by_name(session, role_name)

        if not role:
            role_create = RoleCreate(**role_info)
            role = create_role(session=session, role=role_create)
