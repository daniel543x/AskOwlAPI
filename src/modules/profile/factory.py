"""FastAPI wiring for the profile module."""

from fastapi import Depends
from sqlmodel.ext.asyncio.session import AsyncSession

from ...tools.db import get_session
from ..identity.auth.security import get_current_user
from ..identity.user.model import User
from .model import UserPreferences
from .repository import PreferencesRepository


def get_repository(session: AsyncSession = Depends(get_session)):
    return PreferencesRepository(session)


async def get_optional_user_preferences(
    repo: PreferencesRepository = Depends(get_repository),
    user: User = Depends(get_current_user),
) -> UserPreferences | None:
    """The caller's preference row, or None when unset. Never fails."""
    return await repo.get(user.id)
