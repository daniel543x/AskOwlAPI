import uuid
from typing import Optional

from fastapi import Depends, Query
from langchain_core.language_models import BaseChatModel

from ..identity.auth.security import get_current_user
from .factory import LLMFactory, get_llm_factory
from .repository import LLMRepository, get_repository


# REWRITE THE COMMENT INTO SOMETHING BETTER
#
# Thit is deepends for using in routes only.
# 1. It retrieves the user, factory, and repository.
# 2. It uses the repository to determine which model to select.
# 3. It creates LLM instance and returns it ready to use to the endpoint.
#
async def get_llm_client(
    repo: LLMRepository = Depends(get_repository),
    factory: LLMFactory = Depends(get_llm_factory),
    user=Depends(get_current_user),
    request_model_id: Optional[uuid.UUID] = Query(
        default=None
    ),  # Change to display name?
) -> BaseChatModel:

    user_role_names = [role.name for role in user.roles]

    model_config, provider_config = await repo.resolve_model_config(
        user_id=user.id,
        user_roles=user_role_names,
        requested_model_id=request_model_id,
    )

    llm_client = factory.create_llm(model_config, provider_config)

    return llm_client
