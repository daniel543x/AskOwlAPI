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
    # 6 hours; sessions outlive this via refresh tokens (PLAN Phase 8).
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 360

    # <----- Refresh-token sessions (PLAN Phase 8) ----->
    # Single-use rotation: a session lives this long while it keeps being
    # refreshed; abandoned sessions die with their last token.
    REFRESH_TOKEN_EXPIRE_DAYS: int = 90
    # Superseded (rotated/expired/revoked) rows are kept this long for
    # reuse-detection evidence and audit, then purged by a startup/lifespan
    # sweep — keeps the table small without losing forensic history.
    REFRESH_TOKEN_RETENTION_DAYS: int = 30

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
    # Hard ceiling for the per-request `max_urls` override (WU-6).
    TOP_N_URLS_MAX: int = 20
    TOP_N_CHUNKS: int = 20

    # <----- Search-result hygiene (AR-2 / WU-3) ----->
    # Drop results topically unrelated to the query before they cost a page
    # fetch. On by default: validated by the Phase-3 eval re-run (it drops the
    # exact rust-query -> peptide-articles poisoning class).
    RESULT_SANITY_FILTER: bool = True
    # Fusion floor: drop merged results scoring below this before fetching.
    # 0.0 = off; meaningful only on calibrated fusion scores, so the default
    # stays off until measured on the eval harness.
    URL_SELECTION_MIN_FUSION: float = 0.0
    # Use the fusion score as the cross-encoder tie-breaker. MiniLM scores
    # saturate around 0.999x, making its within-top-N ordering noise; fusion
    # (engine agreement + domain authority + position) breaks those ties.
    FUSION_TIEBREAK: bool = True
    # Domain authority for the fusion score: registrable domains or
    # dot-leading TLD suffixes -> 0..1. Unknown hosts score 0.5.
    DOMAIN_AUTHORITY: dict[str, float] = Field(
        default={
            ".gov": 0.9,
            ".edu": 0.9,
            "wikipedia.org": 0.9,
            "arxiv.org": 0.85,
            "github.com": 0.8,
            "stackoverflow.com": 0.8,
            "postgresql.org": 0.8,
            "python.org": 0.8,
        },
        description=(
            "Domain authority map for search-result fusion scoring. "
            "JSON in the environment variable, e.g. "
            '"{\"github.com\": 0.8, \".gov\": 0.9}".'
        ),
    )

    # <----- Scraping ----->
    SCRAPER_CONCURRENCY: int = 5
    SCRAPER_TIMEOUT: float = 30.0

    # <----- LLM bootstrap defaults ----->
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    DEFAULT_CHAT_MODEL_NAME: str = "qwen3.5"
    MODEL_PROBE_TIMEOUT: float = 3.0

    # Context window (tokens) for Ollama chat models that don't set num_ctx
    # in their catalog default_params. Ollama's own default is small and
    # silently truncates large prompts (e.g. research synthesis), which makes
    # the model drop the tool instruction and answer in prose.
    OLLAMA_NUM_CTX: int = 32768

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

    # <----- Hot-path caches (WU-10 / AR-1) ----->
    # Search-result memoization (per provider config, in-process). A cache
    # hit is invisible except latency; empty results are never cached, so a
    # transient upstream failure is never frozen in.
    SEARCH_CACHE: bool = True
    SEARCH_CACHE_TTL_S: float = 3600.0
    SEARCH_CACHE_MAX_ENTRIES: int = 512
    # Per-URL scraped-content cache, Postgres-backed (survives restarts).
    # A hit skips both the HTTP fetch and the extraction. The `cache:
    # "bypass"` request parameter forces fresh fetches for one turn.
    PAGE_CACHE: bool = True
    PAGE_CACHE_TTL_S: float = 86400.0
    # Rule-based query fan-out ceiling (WU-2; request values clamp to this).
    QUERY_VARIATIONS_MAX: int = 4

    # <----- Shared small-state cache (ICache) ----->
    # In-process memory is the default backend. Setting a Redis URL switches
    # the backend for small TTL state (rate-limit windows; capability probes
    # later). The `redis` package is an optional extra (`uv sync
    # --all-extras`); Redis failures degrade to memory, never crash.
    REDIS_URL: str = ""

    # <----- Decision layer (Phase 7 pre-scores; off by default until eval'd) ----->
    PRE_SCORE_SEARCH_FILTER: bool = False
    PRE_SCORE_CHUNK_FILTER: bool = False
    PRE_SCORE_QUERY_CHOICE: bool = False
    PRE_SCORE_GROUNDING_CHECK: bool = False
    # noul P(relevant) a search result must reach to be fetched.
    PRE_SCORE_RELEVANCE_THRESHOLD: float = 0.5
    # 1-5 chunk relevance floor for the second-opinion filter.
    PRE_SCORE_CHUNK_MIN_SCORE: float = 3.0

    # <----- Outbound proxy ----->
    # Optional HTTP(S) proxy for search + scraping egress. Hosting
    # environments where direct traffic from app hosts is blocked set this
    # to route egress through the provider's proxy. Empty = direct.
    # LLM/embedding provider clients stay direct (often localhost).
    PROXY_URL: str = ""

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


settings = Settings()  # pyright: ignore[reportCallIssue]
