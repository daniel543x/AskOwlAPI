import uuid
from typing import List, Optional, Tuple

from sqlmodel import Session, select

from .model import DefaultSettings, LLMModelCatalog, LLMProvider, UserLLMPreference


class LLMRepository:
    def __init__(self, session: Session):
        self.session = session

    def resolve_model_config(
        self,
        user_id: uuid.UUID,
        user_role: str,
        requested_model_id: Optional[uuid.UUID] = None,
    ) -> Tuple[LLMModelCatalog, LLMProvider]:

        target_model = None

        if requested_model_id:
            target_model = self._get_and_validate_model(requested_model_id, user_role)
            if not target_model:
                raise PermissionError(
                    f"Permission Denied do model ID={requested_model_id} or model not exist."
                )

        if not target_model:
            preference = self.session.get(UserLLMPreference, user_id)
            if preference and preference.selected_model_id:
                target_model = self._get_and_validate_model(
                    preference.selected_model_id, user_role
                )

        if not target_model:
            target_model = self._get_system_default_model(user_role)

        if not target_model:
            raise ValueError("Failed! Miss default model.")

        provider = self.session.get(LLMProvider, target_model.provider_id)

        if not provider:
            raise ValueError(f"Miss config for LLM provider {target_model.model_name}")

        return target_model, provider

    def get_available_models(self, user_role: str) -> List[LLMModelCatalog]:

        statement = select(LLMModelCatalog).where(LLMModelCatalog.is_active)
        models = self.session.exec(statement).all()

        return [
            m for m in models if self._check_role_access(user_role, m.required_role)
        ]

    def set_user_preference(
        self, user_id: uuid.UUID, model_id: uuid.UUID, user_role: str
    ) -> UserLLMPreference:

        model = self._get_and_validate_model(model_id, user_role)
        if not model:
            raise PermissionError("Permission Denied.")

        preference = self.session.get(UserLLMPreference, user_id)
        if not preference:
            preference = UserLLMPreference(user_id=user_id, selected_model_id=model_id)
        else:
            preference.selected_model_id = model_id

        self.session.add(preference)
        self.session.commit()
        self.session.refresh(preference)
        return preference

    def _get_and_validate_model(
        self, model_id: uuid.UUID, user_role: str
    ) -> Optional[LLMModelCatalog]:

        model = self.session.get(LLMModelCatalog, model_id)

        if model and model.is_active:
            if self._check_role_access(user_role, model.required_role):
                return model
        return None

    def _get_system_default_model(self, user_role: str) -> Optional[LLMModelCatalog]:
        setting = self.session.get(DefaultSettings, "default_model_id")
        if setting and setting.value:
            try:
                model_uuid = uuid.UUID(setting.value)
                return self._get_and_validate_model(model_uuid, user_role)
            except ValueError:
                return None
        return None

    def _check_role_access(self, user_role: str, required_role: str) -> bool:
        # user < pro < admin.
        role_hierarchy = {"user": 0, "pro": 1, "admin": 2}

        user_level = role_hierarchy.get(user_role, 0)
        required_level = role_hierarchy.get(required_role, 0)

        return user_level >= required_level
