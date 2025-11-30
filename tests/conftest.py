# Add this import at the very top
from dotenv import load_dotenv

# Load environment variables from .env file immediately
load_dotenv()

import asyncio
import os
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from src.app import app
from src.modules.identity import models
from src.modules.identity.role.model import Role, UserRole

# Import all your models here
from src.modules.identity.user.model import User
from src.tools.db import get_session, init_db  # Check this file as per Step 2

# --- Build the Test Database URL ONCE ---
# Use a single, clear block to build the URL.
# This makes it easy to debug.
POSTGRES_USER = os.getenv("POSTGRES_USER", "user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
POSTGRES_SERVER = os.getenv("POSTGRES_SERVER", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "database")

TEST_DATABASE_URL = (
    f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@"
    f"{POSTGRES_SERVER}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# --- DEBUGGING STEP ---
# Add this print statement to see the EXACT URL being used.
print(f"--- PYTEST: Using Test Database URL: {TEST_DATABASE_URL} ---")

# ... (rest of your conftest.py fixtures are likely fine) ...
test_engine = create_async_engine(TEST_DATABASE_URL, echo=True)
TestSessionLocal = async_sessionmaker(
    bind=test_engine, class_=AsyncSession, expire_on_commit=False
)


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Pytest fixture that provides a transactional scope for a test.
    """
    # --- THIS IS THE CRITICAL DEBUGGING STEP ---
    # Print the names of all tables registered in SQLModel.metadata.
    # This will tell us for sure if the models were imported correctly.
    print("\n--- PYTEST DEBUG: Registered tables in metadata ---")
    for table_name in SQLModel.metadata.tables:
        print(f"  - {table_name}")
    print("--------------------------------------------------\n")

    # The init_db function should now be able to find all tables.
    await init_db(engine=test_engine)

    async with TestSessionLocal() as session:
        yield session

    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    def override_get_session():
        return db_session

    app.dependency_overrides[get_session] = override_get_session

    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()
