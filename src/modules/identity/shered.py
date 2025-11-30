import uuid

from sqlalchemy import PrimaryKeyConstraint
from sqlmodel import Field, SQLModel


class UserRole(SQLModel, table=True):
    __table_args__ = (PrimaryKeyConstraint("user_id", "role_id"),)

    user_id: uuid.UUID = Field(foreign_key="user.id")
    role_id: uuid.UUID = Field(foreign_key="role.id")
