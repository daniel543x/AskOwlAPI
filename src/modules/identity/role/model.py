import uuid
from typing import TYPE_CHECKING

from sqlmodel import Field, Relationship, SQLModel

from ..shered import UserRole

if TYPE_CHECKING:
    from ..user.model import User


class RoleBase(SQLModel):
    name: str = Field(index=True, unique=True)
    description: str | None = Field(default=None, max_length=255)


class Role(RoleBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    # Quoted forward ref: User is only importable under TYPE_CHECKING (cycle).
    users: list["User"] = Relationship(  # noqa: UP037
        back_populates="roles", link_model=UserRole
    )


class RoleCreate(RoleBase):
    pass


class RoleRead(RoleBase):
    id: uuid.UUID
