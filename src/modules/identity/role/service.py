from typing import List
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from .model import Role, RoleCreate, UserRole


async def create_role(session: AsyncSession, role: "RoleCreate") -> Role:
    db_role = Role.model_validate(role)
    session.add(db_role)
    await session.commit()
    await session.refresh(db_role)
    return db_role


async def get_role_by_name(session: AsyncSession, name: str) -> Role | None:
    statement = select(Role).where(Role.name == name)
    result = await session.execute(statement)
    return result.scalar_one_or_none()


async def assign_role_to_user(
    session: AsyncSession, user_id: UUID, role_name: str
) -> bool:
    from ..user.model import User

    user = await session.get(User, user_id)
    role = await get_role_by_name(session, role_name)

    if not user or not role:
        return False

    statement = select(UserRole).where(
        UserRole.user_id == user_id, UserRole.role_id == role.id
    )
    existing_link = (await session.execute(statement)).scalar_one_or_none()
    if existing_link:
        return False

    user_role_link = UserRole(user_id=user_id, role_id=role.id)
    session.add(user_role_link)
    await session.commit()
    return True


async def remove_role_from_user(
    session: AsyncSession, user_id: UUID, role_name: str
) -> None:
    role = await get_role_by_name(session, role_name)
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Role '{role_name}' not found.",
        )

    if role.name.lower() == "admin":
        admin_role_id = role.id
        statement = select(func.count()).where(UserRole.role_id == admin_role_id)
        admin_count_result = await session.execute(statement)
        admin_count = admin_count_result.scalar_one()

        if admin_count <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot remove the last administrator role from the system.",
            )

    statement = select(UserRole).where(
        UserRole.user_id == user_id, UserRole.role_id == role.id
    )
    user_role_link_result = await session.execute(statement)
    user_role_link = user_role_link_result.scalar_one_or_none()

    if not user_role_link:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User does not have this role.",
        )

    await session.delete(user_role_link)
    await session.commit()


async def get_user_roles(session: AsyncSession, user_id: UUID) -> List[str]:
    statement = select(Role.name).join(UserRole).where(UserRole.user_id == user_id)
    result = await session.execute(statement)
    return list(result.scalars().all())


# --- Auxiliary function for creating default roles. ---
async def init_roles(session: AsyncSession) -> None:
    default_roles_data = [
        {"name": "user", "description": "Default user with standard permissions."},
        {"name": "admin", "description": "Administrator with full system access."},
    ]

    for role_info in default_roles_data:
        role_name = role_info["name"]
        role = await get_role_by_name(session, role_name)

        if not role:
            role_create = RoleCreate(**role_info)
            role = await create_role(session=session, role=role_create)
