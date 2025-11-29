import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from pydantic import EmailStr
from sqlmodel import Field, Relationship, SQLModel
from typing_extensions import List

from ..shered import UserRole

if TYPE_CHECKING:
    from ..role.model import Role


# --- DB Tabel Model --- #
class User(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    email: EmailStr = Field(unique=True, index=True)
    nickname: str = Field(unique=True, index=True)
    password_hash: str = Field(..., exclude=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(
        default_factory=datetime.utcnow, sa_column_kwargs={"onupdate": datetime.utcnow}
    )

    roles: List["Role"] = Relationship(back_populates="users", link_model=UserRole)


# --- CRUD Models ---
class UserBase(SQLModel):
    email: EmailStr = Field(unique=True, index=True)


class UserCreate(UserBase):
    nickname: str
    password: str


class UserRead(UserBase):
    id: uuid.UUID
    nickname: str
    created_at: datetime
    updated_at: Optional[datetime] = None


class UserUpdate(SQLModel):
    email: Optional[EmailStr] = None
    nickname: Optional[str] = None
    password: Optional[str] = None
