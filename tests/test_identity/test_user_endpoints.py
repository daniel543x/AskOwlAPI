import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_user(client: AsyncClient):
    user_data = {
        "email": "testUser@test.com",
        "nickname": "testUser",
        "password": "testPassWD",
    }

    # Make a POST request to the user creation endpoint
    response = await client.post("/user/", json=user_data)

    # Assert that the request was successful
    assert response.status_code == 201

    # Assert the response data is correct
    data = response.json()
    assert data["email"] == user_data["email"]
    assert data["nickname"] == user_data["nickname"]
    assert "id" in data
    assert (
        "password" not in data
    )  # Password is not in the response ("exclude=True" for password in user model)
