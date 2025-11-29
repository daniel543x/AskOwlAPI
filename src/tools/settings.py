from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SECRET_KEY: str = "ac258d4cbe17f4f9957d6fafc9edf704f02ede24c339ffd679e25229910ee573"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30


settings = Settings()
