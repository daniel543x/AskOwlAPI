import uuid
from datetime import datetime
from typing import Optional

from pydantic import EmailStr
from sqlmodel import Field, SQLModel


# --- DB Tabel Model --- #
class User(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    email: EmailStr = Field(unique=True, index=True)
    password_hash: str = Field(..., exclude=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(
        default_factory=datetime.utcnow, sa_column_kwargs={"onupdate": datetime.utcnow}
    )


# --- CRUD Models ---
class UserBase(SQLModel):
    email: EmailStr = Field(unique=True, index=True)


class UserCreate(UserBase):
    password: str


class UserRead(UserBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: Optional[datetime] = None


class UserUpdate(SQLModel):
    email: Optional[EmailStr] = None
    password: Optional[str] = None
