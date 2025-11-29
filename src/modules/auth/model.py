from typing import TypedDict

from pydantic import BaseModel


class TokenData(BaseModel):
    access_token: str
    token_type: str


class JWTPayload(TypedDict):
    sub: str
    exp: int


class LoginRequest(BaseModel):
    email: str
    password: str
