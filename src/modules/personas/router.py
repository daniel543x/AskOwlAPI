"""Persona management endpoints (CRUD, own personas only)."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel.ext.asyncio.session import AsyncSession

from ...tools.db import get_session
from ...modules.llm_provider.repository import ModelRepository
from ...modules.llm_provider.selection import unusable_reason
from ..identity.auth.security import get_current_user
from ..identity.user.model import User
from .model import Persona, PersonaCreate, PersonaRead, PersonaUpdate
from .repository import PersonaRepository

router = APIRouter(prefix="/personas", tags=["Personas"])


def get_repository(
    session: AsyncSession = Depends(get_session),
) -> PersonaRepository:
    return PersonaRepository(session)


def get_model_repository(
    session: AsyncSession = Depends(get_session),
) -> ModelRepository:
    return ModelRepository(session)


async def _validate_persona_model(
    repo: ModelRepository, user: User, model_id: UUID | None
) -> None:
    """Reject persona model ids that could never run an agent turn."""
    if model_id is None:
        return
    state = await repo.entry_state(model_id)
    reason = unusable_reason(state, [role.name for role in user.roles])
    if reason is not None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"code": "persona_model_invalid", "reason": reason},
        )


@router.post("", response_model=PersonaRead, status_code=status.HTTP_201_CREATED)
async def create_persona(
    data: PersonaCreate,
    repo: PersonaRepository = Depends(get_repository),
    model_repo: ModelRepository = Depends(get_model_repository),
    user: User = Depends(get_current_user),
):
    await _validate_persona_model(model_repo, user, data.model)
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
    model_repo: ModelRepository = Depends(get_model_repository),
    user: User = Depends(get_current_user),
):
    persona = await repo.get_for_user(user.id, persona_id)
    if persona is None:
        raise HTTPException(status_code=404, detail="Persona not found.")
    await _validate_persona_model(model_repo, user, changes.model)
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
