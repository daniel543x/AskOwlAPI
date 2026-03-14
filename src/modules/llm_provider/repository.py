import uuid
from typing import List, Optional, Tuple

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from ...tools.db import get_session
from .model import DefaultSettings, LLMModelCatalog, LLMProvider, UserLLMPreference


class LLMRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def resolve_model_config(
        self,
        user_id: uuid.UUID,
        user_role: str,
        requested_model_id: Optional[uuid.UUID] = None,
    ) -> Tuple[LLMModelCatalog, LLMProvider]:

        target_model = None

        if requested_model_id:
            target_model = await self._get_and_validate_model(
                requested_model_id, user_role
            )
            if not target_model:
                raise PermissionError(
                    f"Permission Denied model ID={requested_model_id} or model not exist."
                )

        if not target_model:
            preference = await self.session.get(UserLLMPreference, user_id)
            if preference and preference.selected_model_id:
                target_model = await self._get_and_validate_model(
                    preference.selected_model_id, user_role
                )

        if not target_model:
            target_model = await self._get_system_default_model(user_role)

        if not target_model:
            raise ValueError("Brak skonfigurowanego modelu domyślnego w systemie.")

        provider = await self.session.get(LLMProvider, target_model.provider_id)

        if not provider:
            raise ValueError(
                f"Brak konfiguracji dostawcy dla modelu {target_model.model_name}"
            )

        return target_model, provider

    async def get_available_models(self, user_role: str) -> List[LLMModelCatalog]:
        statement = select(LLMModelCatalog).where(LLMModelCatalog.is_active)

        result = await self.session.execute(statement)
        models = result.scalars().all()

        return [
            m for m in models if self._check_role_access(user_role, m.required_role)
        ]

    async def set_user_preference(
        self, user_id: uuid.UUID, model_id: uuid.UUID, user_role: str
    ) -> UserLLMPreference:
        model = await self._get_and_validate_model(model_id, user_role)
        if not model:
            raise PermissionError("Permission Denied.")

        preference = await self.session.get(UserLLMPreference, user_id)
        if not preference:
            preference = UserLLMPreference(user_id=user_id, selected_model_id=model_id)
        else:
            preference.selected_model_id = model_id

        self.session.add(preference)
        await self.session.commit()
        await self.session.refresh(preference)
        return preference

    async def _get_and_validate_model(
        self, model_id: uuid.UUID, user_role: str
    ) -> Optional[LLMModelCatalog]:
        model = await self.session.get(LLMModelCatalog, model_id)

        if model and model.is_active:
            if self._check_role_access(user_role, model.required_role):
                return model
        return None

    async def _get_system_default_model(
        self, user_role: str
    ) -> Optional[LLMModelCatalog]:
        setting = await self.session.get(DefaultSettings, "default_model_id")
        if setting and setting.value:
            try:
                model_uuid = uuid.UUID(setting.value)
                return await self._get_and_validate_model(model_uuid, user_role)
            except ValueError:
                return None
        return None

    def _check_role_access(self, user_role: str, required_role: str) -> bool:
        role_hierarchy = {"user": 0, "pro": 1, "admin": 2}
        user_level = role_hierarchy.get(user_role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        return user_level >= required_level


# Deppendency
def get_repository(session: AsyncSession = Depends(get_session)) -> LLMRepository:
    return LLMRepository(session)
