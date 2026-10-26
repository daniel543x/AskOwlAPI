# Configuration

All configuration goes through `src/tools/settings.py` (pydantic-settings):
environment variables and the `.env` file. See `.env.template` for the full
annotated list. Nothing is hardcoded; the app refuses to start without a
valid `MASTER_KEY` (Fernet key, url-safe base64, 32 bytes).

## Required

| Variable | Notes |
|---|---|
| `POSTGRES_SERVER/USER/PASSWORD/DB/PORT` | database connection |
| `MASTER_KEY` | encrypts provider API keys at rest; also JWT signing key |
| `SEARXNG_SECRET` | required by the compose-managed SearXNG |

## Optional (defaults shown)

| Group | Variables |
|---|---|
| Logging | `LOG_LEVEL=INFO`, `DEBUG=false` (per-stage tracing, `debug` SSE events) |
| Search | `SEARXNG_BASE_URL`, `SEARXNG_ENGINES`, `SEARXNG_CATEGORIES`, `SEARXNG_TIMEOUT` |
| Retrieval | `TOP_N_URLS=10`, `TOP_N_CHUNKS=20`, `VECTOR_CANDIDATES_K=25`, `RANKER_MODEL_NAME`, `RANKER_CACHE_DIR` |
| Scraping | `SCRAPER_CONCURRENCY=5`, `SCRAPER_TIMEOUT=30` |
| Model defaults | `OLLAMA_BASE_URL`, `DEFAULT_CHAT_MODEL_NAME=qwen3.5`, `DEFAULT_EMBEDDING_MODEL_NAME=bge-m3`, `DEFAULT_EMBEDDING_MODEL_DIMS=1024` |
| Research/agent | `RESEARCH_MAX_SUBQUESTIONS=5`, `AGENT_MAX_ITERATIONS=8`, `AGENT_TIME_BUDGET_SECONDS=300`, `AGENT_TOKEN_BUDGET=60000` |
| API keys | `API_KEY_RATE_LIMIT_PER_MINUTE=60` |
| Egress proxy | `PROXY_URL` (empty = direct) |
| Decision pre-scores | `PRE_SCORE_SEARCH_FILTER/CHUNK_FILTER/QUERY_CHOICE/GROUNDING_CHECK` (all off), `PRE_SCORE_RELEVANCE_THRESHOLD`, `PRE_SCORE_CHUNK_MIN_SCORE` — see [../architecture/decision_layer.md](../architecture/decision_layer.md) |
| Auth | `ACCESS_TOKEN_EXPIRE_MINUTES=360` (6 h), `REFRESH_TOKEN_EXPIRE_DAYS=90`, `REFRESH_TOKEN_RETENTION_DAYS=30` — see [../architecture/auth.md](../architecture/auth.md) |

Model selection itself is **not** env-driven: it goes through the
admin-curated catalog (`/models` routes), per PLAN product decision §0.1.

## Outbound proxy

Hosting providers often block or rate-limit direct traffic from app hosts
(search engines return CAPTCHAs, pages answer 403/429). Set `PROXY_URL`
(e.g. `http://proxy.internal:8080`) and **all search-engine and
page-fetching egress** routes through it — one setting, applied by the
shared client factory (`src/tools/http.py`). LLM/embedding provider
clients stay direct because they frequently target localhost or the LAN;
route those via their provider base URLs if ever needed.

## Security rules

- Never commit `.env` or real keys.
- Provider API keys are entered via the admin API and stored
  Fernet-encrypted; user API keys are stored as SHA-256 hashes.
- `DEBUG=true` traces pipeline stages and LLM tool-call payloads with API
  keys redacted; keep it off in production.
