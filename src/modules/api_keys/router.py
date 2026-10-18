"""API-key management endpoints (own keys only; raw key shown once)."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ...tools.db import get_session
from ..identity.auth.security import get_current_user
from ..identity.user.model import User
from .model import ApiKeyCreate, ApiKeyCreated, ApiKeyRead
from .repository import ApiKeyRepository

router = APIRouter(prefix="/api-keys", tags=["API Keys"])


def get_repository(
    session: AsyncSession = Depends(get_session),
) -> ApiKeyRepository:
    return ApiKeyRepository(session)


@router.post("", response_model=ApiKeyCreated, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    data: ApiKeyCreate,
    repo: ApiKeyRepository = Depends(get_repository),
    user: User = Depends(get_current_user),
):
    try:
        api_key, raw = await repo.create(user.id, data)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    payload = ApiKeyRead.model_validate(api_key, from_attributes=True)
    return ApiKeyCreated(**payload.model_dump(), raw_key=raw)


@router.get("", response_model=list[ApiKeyRead])
async def list_api_keys(
    repo: ApiKeyRepository = Depends(get_repository),
    user: User = Depends(get_current_user),
):
    keys = await repo.list_for_user(user.id)
    return [ApiKeyRead.model_validate(k, from_attributes=True) for k in keys]


@router.delete("/{api_key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    api_key_id: UUID,
    repo: ApiKeyRepository = Depends(get_repository),
    user: User = Depends(get_current_user),
):
    api_key = await repo.get_for_user(user.id, api_key_id)
    if api_key is None:
        raise HTTPException(status_code=404, detail="API key not found.")
    await repo.revoke(api_key)
