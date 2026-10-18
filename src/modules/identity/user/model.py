import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import EmailStr
from sqlmodel import Field, Relationship, SQLModel

from ....tools.settings import settings
from ..shered import UserRole

if TYPE_CHECKING:
    from ..role.model import Role


# --- DB Tabel Model --- #
class User(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    email: EmailStr = Field(unique=True, index=True)
    nickname: str = Field(
        unique=True, index=True, max_length=settings.USERNAME_CHAR_LIMIT
    )
    password_hash: str = Field(..., exclude=True)
    is_active: bool = Field(default=True, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime | None = Field(
        default_factory=datetime.utcnow, sa_column_kwargs={"onupdate": datetime.utcnow}
    )

    # Quoted forward ref: Role is only importable under TYPE_CHECKING (cycle).
    roles: list["Role"] = Relationship(  # noqa: UP037
        back_populates="users",
        link_model=UserRole,
        sa_relationship_kwargs={"lazy": "selectin"},
    )


# --- CRUD Models ---
class UserBase(SQLModel):
    email: EmailStr = Field(unique=True, index=True)
    nickname: str = Field(
        unique=True, index=True, max_length=settings.USERNAME_CHAR_LIMIT
    )


class UserCreate(UserBase):
    password: str


class UserRead(UserBase):
    id: uuid.UUID
    is_active: bool
    created_at: datetime
    updated_at: datetime | None = None


class UserUpdate(SQLModel):
    email: EmailStr | None = None
    nickname: str | None = None
    password: str | None = None


class UserAdminUpdate(SQLModel):
    """Admin-only mutable fields (block/unblock)."""

    is_active: bool | None = None
