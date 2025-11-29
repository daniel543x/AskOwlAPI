from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from ...tools.db import get_session
from ..auth.security import get_current_admin_user
from . import service
from .model import Role, RoleCreate, RoleRead

router = APIRouter(
    prefix="/roles", tags=["Roles"], dependencies=[Depends(get_current_admin_user)]
)


@router.post("/", response_model=RoleRead, status_code=status.HTTP_201_CREATED)
def create_role(*, session: Session = Depends(get_session), role_in: RoleCreate):
    if service.get_role_by_name(session, role_in.name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Role with this name already exists",
        )

    return service.create_role(session=session, role=role_in)


@router.get("/", response_model=List[RoleRead])
def list_roles(session: Session = Depends(get_session)):
    roles = session.exec(select(Role)).all()
    return roles


@router.post("/users/{user_id}/assign")
def assign_role_to_user(
    user_id: UUID, role_name: str, session: Session = Depends(get_session)
):
    if not service.assign_role_to_user(
        session=session, user_id=user_id, role_name=role_name
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User or role not found, or user already has this role.",
        )
    return {"message": f"Role '{role_name}' assigned to user {user_id} successfully."}


@router.delete("/users/{user_id}/remove")
def remove_role_from_user(
    user_id: UUID, role_name: str, session: Session = Depends(get_session)
):
    service.remove_role_from_user(session=session, user_id=user_id, role_name=role_name)
    return {"message": f"Role '{role_name}' removed from user {user_id} successfully."}
