import logging
import uuid

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from ...tools.db import get_session
from ...tools.settings import settings
from .model import DefaultSettings, LLMModelCatalog, LLMProvider, UserLLMPreference

logger = logging.getLogger(__name__)


class LLMRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def resolve_model_config(
        self,
        user_id: uuid.UUID,
        user_roles: list[str],
        requested_model_id: uuid.UUID | None = None,
    ) -> tuple[LLMModelCatalog, LLMProvider]:

        target_model = None

        if requested_model_id:
            target_model = await self._get_and_validate_model(
                requested_model_id, user_roles
            )
            if not target_model:
                raise PermissionError(
                    f"Permission Denied model ID={requested_model_id} "
                    f"or model not exist."
                )

        if not target_model:
            preference = await self.session.get(UserLLMPreference, user_id)
            if preference and preference.selected_model_id:
                target_model = await self._get_and_validate_model(
                    preference.selected_model_id, user_roles
                )

        if not target_model:
            target_model = await self._get_system_default_model(user_roles)

        if not target_model:
            raise ValueError("Not found model config.")

        provider = await self.session.get(LLMProvider, target_model.provider_id)

        if not provider:
            raise ValueError(
                f"Not found config for model provider: {target_model.model_name}"
            )

        return target_model, provider

    async def get_available_models(
        self, user_roles: list[str]
    ) -> list[LLMModelCatalog]:
        statement = select(LLMModelCatalog).where(LLMModelCatalog.is_active)

        result = await self.session.execute(statement)
        models = result.scalars().all()

        return [
            m for m in models if self._check_role_access(user_roles, m.required_role)
        ]

    async def set_user_preference(
        self, user_id: uuid.UUID, model_id: uuid.UUID, user_role: list[str]
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
        self, model_id: uuid.UUID, user_roles: list[str]
    ) -> LLMModelCatalog | None:
        model = await self.session.get(LLMModelCatalog, model_id)

        if model and model.is_active:
            if self._check_role_access(user_roles, model.required_role):
                return model
        return None

    async def _get_system_default_model(
        self, user_roles: list[str]
    ) -> LLMModelCatalog | None:
        setting = await self.session.get(DefaultSettings, "default_model_id")
        if setting and setting.value:
            try:
                model_uuid = uuid.UUID(setting.value)
                return await self._get_and_validate_model(model_uuid, user_roles)
            except ValueError:
                return None
        return None

    def _check_role_access(self, user_roles: list[str], required_role: str) -> bool:
        role_hierarchy = {"user": 0, "pro": 1, "admin": 2}
        max_user_level = 0
        for role in user_roles:
            level = role_hierarchy.get(role, 0)
            if level > max_user_level:
                max_user_level = level
        required_level = role_hierarchy.get(required_role, 0)
        return max_user_level >= required_level

    # --------------------------------------------------------- #

    async def init_default_llm_config(self) -> None:
        statement = select(LLMProvider).where(LLMProvider.name == "Local Ollama")
        result = await self.session.execute(statement)
        provider = result.scalar_one_or_none()

        if not provider:
            logger.info("Seeding provider 'Local Ollama'...")
            provider = LLMProvider(
                name="Local Ollama",
                provider_type="ollama",
                base_url=settings.OLLAMA_BASE_URL,
                api_key_encrypted=None,
            )
            self.session.add(provider)
            await self.session.flush()
        else:
            logger.debug("Provider 'Local Ollama' already exists.")

        statement = select(LLMModelCatalog).where(
            LLMModelCatalog.model_name == settings.DEFAULT_CHAT_MODEL_NAME,
            LLMModelCatalog.provider_id == provider.id,
        )
        result = await self.session.execute(statement)
        model = result.scalar_one_or_none()

        if not model:
            logger.info("Seeding model '%s'...", settings.DEFAULT_CHAT_MODEL_NAME)
            model = LLMModelCatalog(
                provider_id=provider.id,
                model_name=settings.DEFAULT_CHAT_MODEL_NAME,
                display_name=f"{settings.DEFAULT_CHAT_MODEL_NAME} (Local)",
                default_params={"temperature": 0.7},
                is_active=True,
                required_role="user",
            )
            self.session.add(model)
            await self.session.flush()
        else:
            logger.debug("Model '%s' already exists.", settings.DEFAULT_CHAT_MODEL_NAME)

        setting = await self.session.get(DefaultSettings, "default_model_id")

        if not setting:
            logger.info("Setting default model in system settings...")
            setting = DefaultSettings(key="default_model_id", value=str(model.id))
            self.session.add(setting)

        await self.session.commit()
        logger.info("LLM bootstrap finished; default model is ready to use.")


# Deppendency
def get_repository(session: AsyncSession = Depends(get_session)) -> LLMRepository:
    return LLMRepository(session)
