"""Persona management endpoints (CRUD, own personas only)."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ...tools.db import get_session
from ..identity.auth.security import get_current_user
from ..identity.user.model import User
from .model import Persona, PersonaCreate, PersonaRead, PersonaUpdate
from .repository import PersonaRepository

router = APIRouter(prefix="/personas", tags=["Personas"])


def get_repository(
    session: AsyncSession = Depends(get_session),
) -> PersonaRepository:
    return PersonaRepository(session)


@router.post("", response_model=PersonaRead, status_code=status.HTTP_201_CREATED)
async def create_persona(
    data: PersonaCreate,
    repo: PersonaRepository = Depends(get_repository),
    user: User = Depends(get_current_user),
):
    persona = Persona(user_id=user.id, **data.model_dump())
    return await repo.create(persona)


@router.get("", response_model=list[PersonaRead])
async def list_personas(
    repo: PersonaRepository = Depends(get_repository),
    user: User = Depends(get_current_user),
):
    return await repo.list_for_user(user.id)


@router.get("/{persona_id}", response_model=PersonaRead)
async def get_persona(
    persona_id: UUID,
    repo: PersonaRepository = Depends(get_repository),
    user: User = Depends(get_current_user),
):
    persona = await repo.get_for_user(user.id, persona_id)
    if persona is None:
        raise HTTPException(status_code=404, detail="Persona not found.")
    return persona


@router.patch("/{persona_id}", response_model=PersonaRead)
async def update_persona(
    persona_id: UUID,
    changes: PersonaUpdate,
    repo: PersonaRepository = Depends(get_repository),
    user: User = Depends(get_current_user),
):
    persona = await repo.get_for_user(user.id, persona_id)
    if persona is None:
        raise HTTPException(status_code=404, detail="Persona not found.")
    return await repo.update(persona, changes)


@router.delete("/{persona_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_persona(
    persona_id: UUID,
    repo: PersonaRepository = Depends(get_repository),
    user: User = Depends(get_current_user),
):
    persona = await repo.get_for_user(user.id, persona_id)
    if persona is None:
        raise HTTPException(status_code=404, detail="Persona not found.")
    await repo.delete(persona)
