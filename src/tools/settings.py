import logging

from cryptography.fernet import Fernet
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration.

    Values are read from environment variables and the local `.env` file.
    `MASTER_KEY` is required: the application refuses to start without it.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # <----- Security ----->
    MASTER_KEY: str = Field(
        description="Fernet key used for secret encryption. Required, no fallback."
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # <----- Logging ----->
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = False

    # <----- DB ----->
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "postgres"
    POSTGRES_PORT: int = 5432

    # <----- User ----->
    USERNAME_CHAR_LIMIT: int = 50

    # <----- Search engine (SearXNG) ----->
    SEARXNG_BASE_URL: str = "http://localhost:8080"
    SEARXNG_ENGINES: str = "bing,brave,duckduckgo,google,qwant,startpage"
    SEARXNG_CATEGORIES: str = "general"
    SEARXNG_TIMEOUT: float = 60.0

    # <----- Ranking ----->
    RANKER_MODEL_NAME: str = "ms-marco-MiniLM-L-12-v2"
    RANKER_CACHE_DIR: str = "/tmp/cache"
    TOP_N_URLS: int = 10
    TOP_N_CHUNKS: int = 20

    # <----- Scraping ----->
    SCRAPER_CONCURRENCY: int = 5
    SCRAPER_TIMEOUT: float = 30.0

    # <----- LLM bootstrap defaults ----->
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    DEFAULT_CHAT_MODEL_NAME: str = "qwen3.5"
    MODEL_PROBE_TIMEOUT: float = 3.0

    # Default embedding model (PLAN decision #1, 2026-10-17: bge-m3).
    DEFAULT_EMBEDDING_MODEL_NAME: str = "bge-m3"
    DEFAULT_EMBEDDING_MODEL_DIMS: int = 1024

    # <----- Hybrid retrieval ----->
    # Vector candidates fetched from the chunk store before cross-encoder
    # reranking. Chunks are embedded at scrape time when an embedder resolves.
    VECTOR_CANDIDATES_K: int = 25

    # <----- Research mode / agent loop ----->
    RESEARCH_MAX_SUBQUESTIONS: int = 5
    AGENT_MAX_ITERATIONS: int = 8
    AGENT_TIME_BUDGET_SECONDS: float = 300.0
    # Estimated tokens (chars/4) across all LLM calls of one research/agent run.
    AGENT_TOKEN_BUDGET: int = 60_000

    # <----- API keys ----->
    # Per-key fixed-window limit for X-API-Key traffic (requests/minute).
    API_KEY_RATE_LIMIT_PER_MINUTE: int = 60

    @field_validator("MASTER_KEY")
    @classmethod
    def _validate_master_key(cls, value: str) -> str:
        try:
            Fernet(value.encode("ascii"))
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "MASTER_KEY must be a valid Fernet key (url-safe base64, 32 bytes). "
                'Generate one with: python -c "from cryptography.fernet import Fernet; '
                'print(Fernet.generate_key().decode())"'
            ) from exc
        return value

    def resolved_log_level(self) -> int:
        if self.DEBUG:
            return logging.DEBUG
        return logging.getLevelNamesMapping().get(
            self.LOG_LEVEL.upper(), logging.INFO
        )


settings = Settings()
