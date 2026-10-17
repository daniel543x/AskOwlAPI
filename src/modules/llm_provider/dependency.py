"""FastAPI dependencies resolving models from the registry for routes.

1. Resolve the model (request param -> user preference -> global default ->
   built-in local default) with role checks.
2. Build the client for the model's kind.
3. Return it ready to use.
"""

import uuid

from fastapi import Depends, HTTPException, Query, status
from langchain_core.language_models import BaseChatModel

from ..identity.auth.security import get_current_user
from ..identity.user.model import User
from .factory import ChatModelFactory, get_llm_factory
from .model import ModelKind
from .repository import ModelRepository, get_repository
from .selection import SelectionError


def _to_http_error(error: SelectionError) -> HTTPException:
    if error.tier == "permission":
        return HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=error.reason
        )
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=error.reason
    )


def _user_role_names(user: User) -> list[str]:
    return [role.name for role in user.roles]


async def get_llm_client(
    repo: ModelRepository = Depends(get_repository),
    factory: ChatModelFactory = Depends(get_llm_factory),
    user: User = Depends(get_current_user),
    request_model_id: uuid.UUID | None = Query(default=None),
) -> BaseChatModel:
    try:
        resolved = await repo.resolve_model(
            kind=ModelKind.CHAT.value,
            user_id=user.id,
            user_roles=_user_role_names(user),
            requested_model_id=request_model_id,
        )
    except SelectionError as e:
        raise _to_http_error(e) from e

    return factory.create_chat_model(resolved)
