import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # <----- Security ----->
    SECRET_KEY: str = os.getenv(
        "MASTER_KEY", "ac258d4cbe17f4f9957d6fafc9edf704f02ede24c339ffd679e25229910ee573"
    )  # It should be overwrite
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

    # <----- DB ----->
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "localhost")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "postgres")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")

    # <----- User ----->
    USERNAME_CHAR_LIMIT: int = 50


settings = Settings()
