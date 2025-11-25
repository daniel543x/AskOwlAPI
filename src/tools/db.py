import os
from typing import Generator

from sqlmodel import Session, SQLModel, create_engine

POSTGRES_SERVER = os.getenv("POSTGRES_SERVER", "localhost")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

# Setup DataBase
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}:{POSTGRES_PORT}/{POSTGRES_DB}"
engine = create_engine(DATABASE_URL, echo=True)  # echo=True to see generated SQL
SQLModel.metadata.create_all(engine)


def init_table() -> None:
    SQLModel.metadata.create_all(engine)


# For Dependency
def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session
