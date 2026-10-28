import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..identity.auth.security import get_current_admin_user, get_current_user
from ..identity.user.model import User
from .model import (
    DefaultModelRead,
    ModelEntryCreate,
    ModelEntryRead,
    ModelEntryUpdate,
    ModelKind,
    ModelWithProviderRead,
    PreferenceRead,
    ProviderCreate,
    ProviderRead,
    ProviderUpdate,
)
from .repository import ModelRepository, get_repository
from .selection import SelectionError

router = APIRouter(tags=["Models"])


def _provider_read(provider) -> ProviderRead:
    return ProviderRead(
        id=provider.id,
        name=provider.name,
        provider_type=provider.provider_type,
        base_url=provider.base_url,
        is_active=provider.is_active,
        health_status=provider.health_status,
        last_probe_at=provider.last_probe_at,
        has_api_key=bool(provider.api_key_encrypted),
    )


def _entry_read(entry, provider=None) -> ModelWithProviderRead:
    return ModelWithProviderRead(
        id=entry.id,
        provider_id=entry.provider_id,
        kind=entry.kind,
        model_name=entry.model_name,
        display_name=entry.display_name,
        is_active=entry.is_active,
        required_role=entry.required_role,
        capabilities=list(entry.capabilities or []),
        effort_levels=list(entry.effort_levels or []),
        vector_dims=entry.vector_dims,
        provider_name=provider.name if provider else None,
        provider_type=provider.provider_type if provider else None,
    )


# <----- User-facing ----->


@router.get("/models", response_model=list[ModelWithProviderRead])
async def list_models(
    kind: ModelKind | None = Query(default=None),
    repo: ModelRepository = Depends(get_repository),
    user: User = Depends(get_current_user),
):
    """List catalog entries available to the calling user (role-filtered)."""
    role_names = [role.name for role in user.roles]
    rows = await repo.get_available_models(role_names, kind.value if kind else None)
    return [_entry_read(entry, provider) for entry, provider in rows]


@router.get("/models/preferences", response_model=list[PreferenceRead])
async def get_my_preferences(
    repo: ModelRepository = Depends(get_repository),
    user: User = Depends(get_current_user),
):
    preferences = await repo.get_user_preferences(user.id)
    return [
        PreferenceRead(
            user_id=p.user_id, kind=p.kind, selected_model_id=p.selected_model_id
        )
        for p in preferences
    ]


@router.put("/models/{model_id}/preference", response_model=PreferenceRead)
async def set_model_preference(
    model_id: uuid.UUID,
    repo: ModelRepository = Depends(get_repository),
    user: User = Depends(get_current_user),
):
    """Set the calling user's default model for the entry's kind."""
    role_names = [role.name for role in user.roles]
    try:
        preference = await repo.set_user_preference(user.id, model_id, role_names)
    except SelectionError as e:
        code = (
            status.HTTP_403_FORBIDDEN
            if e.tier == "permission"
            else status.HTTP_404_NOT_FOUND
        )
        raise HTTPException(status_code=code, detail=e.reason) from e
    return PreferenceRead(
        user_id=preference.user_id,
        kind=preference.kind,
        selected_model_id=preference.selected_model_id,
    )


# <----- Health ----->


health_router = APIRouter(tags=["Models"])


@health_router.get(
    "/health/models",
    response_model=list[ProviderRead],
    dependencies=[Depends(get_current_admin_user)],
)
async def models_health(repo: ModelRepository = Depends(get_repository)):
    """Re-probe every provider and return fresh health status (admin only)."""
    return [_provider_read(p) for p in await repo.probe_providers()]


# <----- Admin: providers / entries / defaults ----->

admin_router = APIRouter(
    prefix="/admin/models",
    tags=["Models"],
    dependencies=[Depends(get_current_admin_user)],
)


@admin_router.post("/providers", response_model=ProviderRead, status_code=201)
async def create_provider(
    provider_in: ProviderCreate, repo: ModelRepository = Depends(get_repository)
):
    try:
        provider = await repo.create_provider(provider_in)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(e)
        ) from e
    return _provider_read(provider)


@admin_router.get("/providers", response_model=list[ProviderRead])
async def list_providers(repo: ModelRepository = Depends(get_repository)):
    return [_provider_read(p) for p in await repo.list_providers()]


@admin_router.patch("/providers/{provider_id}", response_model=ProviderRead)
async def update_provider(
    provider_id: uuid.UUID,
    provider_in: ProviderUpdate,
    repo: ModelRepository = Depends(get_repository),
):
    try:
        provider = await repo.update_provider(provider_id, provider_in)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(e)
        ) from e
    return _provider_read(provider)


@admin_router.delete("/providers/{provider_id}", status_code=204)
async def delete_provider(
    provider_id: uuid.UUID, repo: ModelRepository = Depends(get_repository)
):
    try:
        await repo.delete_provider(provider_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(e)
        ) from e


@admin_router.post("/entries", response_model=ModelEntryRead, status_code=201)
async def create_model_entry(
    entry_in: ModelEntryCreate, repo: ModelRepository = Depends(get_repository)
):
    try:
        entry = await repo.create_model(entry_in)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e
    return _entry_read(entry)


@admin_router.get("/entries", response_model=list[ModelEntryRead])
async def list_model_entries(
    kind: ModelKind | None = Query(default=None),
    repo: ModelRepository = Depends(get_repository),
):
    entries = await repo.list_models(kind.value if kind else None)
    return [_entry_read(e) for e in entries]


@admin_router.patch("/entries/{model_id}", response_model=ModelEntryRead)
async def update_model_entry(
    model_id: uuid.UUID,
    entry_in: ModelEntryUpdate,
    repo: ModelRepository = Depends(get_repository),
):
    try:
        entry = await repo.update_model(model_id, entry_in)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e
    return _entry_read(entry)


@admin_router.delete("/entries/{model_id}", status_code=204)
async def delete_model_entry(
    model_id: uuid.UUID, repo: ModelRepository = Depends(get_repository)
):
    try:
        await repo.delete_model(model_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(e)
        ) from e


@admin_router.put("/defaults/{kind}", response_model=DefaultModelRead)
async def set_default_model(
    kind: ModelKind,
    model_id: uuid.UUID,
    repo: ModelRepository = Depends(get_repository),
):
    try:
        row = await repo.set_default(kind.value, model_id)
        entry = await repo.get_model(model_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e
    return DefaultModelRead(
        kind=kind.value, model_id=uuid.UUID(row.value), model_name=entry.model_name
    )


@admin_router.get("/defaults")
async def get_default_models(repo: ModelRepository = Depends(get_repository)):
    defaults = await repo.get_defaults()
    return {"defaults": {kind: str(model_id) for kind, model_id in defaults.items()}}
