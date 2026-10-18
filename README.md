# AskOwl

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
![Status: WIP](https://img.shields.io/badge/Status-WIP-yellow.svg)

**AskOwl** is a self-hosted, LLM-powered search engine: a FastAPI backend that takes a
question, searches the web (SearXNG), fetches and ranks pages with hybrid retrieval
(pgvector embeddings + cross-encoder reranking), and synthesizes a cited answer with an
LLM. It serves interactive clients (SSE streaming) and other backends (JSON) equally.
Core rule: **the LLM is never used without a tool call** — every reasoning step goes
through native tool calling.

> **Note:** This project is in active development. See `PLAN.md` for the phase status.
> Until the first clients appear, the SSE protocol and route semantics may change.

## Why AskOwl?

Existing AI tools often offer limited personalization. AskOwl was born from the need for
a search engine that adapts to you, not the other way around: programming answers biased
toward your tech stack, or research scoped to your domain. It is self-hosted — full
control over your data, models, and infrastructure; nothing leaves your network unless
you configure a cloud provider.

## Key Features

*   **Self-hosted:** runs on your hardware; models may be local (Ollama), on your
    LAN/host, or in the cloud — any mix.
*   **Admin-curated model catalog:** admins register providers and models (chat,
    embedding, reranker) with encrypted API keys; users pick from what is available to
    their role, with sensible defaults.
*   **Hybrid retrieval:** pages are chunked and embedded (`bge-m3`) into a pgvector
    store; vector candidates are reranked by a cross-encoder (`flashrank`) before the
    answer step.
*   **Two client protocols:** token-by-token SSE streaming and single-document JSON —
    same data, same route.
*   **Search modes:** `search` (direct cited answers), `research` (planner →
    sub-questions → cited report), and `agent` (autonomous tool loop) — all
    bounded by configurable iteration/time/token budgets. See `PLAN.md` for
    the roadmap (personas are next).

## Quick Start

Prerequisites:

*   [Podman](https://podman.io/) or [Docker](https://www.docker.com/)
*   `podman-compose` or `docker-compose`

### Step 1: Clone the repository

```bash
git clone https://github.com/daniel543x/AskOwlAPI.git
cd AskOwlAPI
```

### Step 2: Configure environment variables

```bash
cp .env.template .env
```

Open `.env` and set the variables marked `# required`: Postgres credentials, a
`MASTER_KEY` (Fernet key that encrypts provider API keys at rest — generate one with
`python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`),
and `SEARXNG_SECRET`.

### Step 3: Run with Compose

```bash
# db (pgvector) + searXNG only:
podman-compose up -d
# full stack including the API container:
podman-compose --profile api up -d
# optionally add a local Ollama container:
podman-compose --profile ollama up -d
```

The API listens on `http://localhost:3000/` (interactive docs at `/docs`). The `ollama` profile ships a local model server; pull the default models once:

```bash
podman-compose exec ollama ollama pull qwen3.5
podman-compose exec ollama ollama pull bge-m3
```

### Step 4: Ask a question

```bash
curl -N "http://localhost:3000/ask/search?query=what%20is%20pgvector&stream=true"
# or single JSON document:
curl "http://localhost:3000/ask/search?query=what%20is%20pgvector&stream=false"
```

The SSE protocol (events `status`, `query`, `sources`, `context`, `message`, `done`,
`error`) is documented in
[`docs/api/ask_search_sse.md`](docs/api/ask_search_sse.md); the research/agent
extensions (`plan` event, budget-aware `done`) in
[`docs/api/ask_research_sse.md`](docs/api/ask_research_sse.md).

## Search Modes

All modes require authentication (JWT bearer or `X-API-Key`).

*   **`search`** (all users): direct, enhanced search. Query rewriting, search,
    scraping, hybrid retrieval, and cited answer synthesis — each step a native
    tool call.
*   **`research`** (elevated role `pro`/`admin`): the planner decomposes the
    query into sub-questions via a tool call, runs the search pipeline for
    each, and synthesizes a cited Markdown report. SSE or JSON.
*   **`agent`** (elevated role): an autonomous agent loop over the shared tools
    (`web_search`, `fetch_pages`, `get_context`), bounded by
    iteration/time/token budgets, answering through a `SubmitOutcome` tool
    call. Optionally scoped by one of your **personas** (`persona_id`):
    user-created profiles whose topic context biases queries and whose
    domain/engine constraints scope the search. Manage personas via
    `POST/GET/PATCH/DELETE /personas`.

## Models and Providers

Model selection goes through the admin-curated catalog, not environment variables:

*   Providers are registered over the admin API with a base URL and an API key that is
    stored **encrypted** in the database (keyed by `MASTER_KEY`) and never returned or
    logged. Cloud (e.g. OpenRouter), host, LAN, and local Ollama endpoints are equal
    citizens.
*   Catalog entries carry capabilities (e.g. `tools`, `streaming`); steps that need tool
    calling reject models without the `tools` capability with a clear error.
*   Resolution order per request: explicit `model_id` → user preference (role-checked) →
    global default → built-in local default (Ollama `qwen3.5` for chat, `bge-m3` for
    embeddings).
*   `GET /health/models` reports live provider health; at startup the app probes
    configured providers and seeds the catalog only with what responds.

## Configuration

All configuration lives in `.env` (via pydantic-settings; see `.env.template`): Postgres
connection, `MASTER_KEY`, logging (`LOG_LEVEL`, `DEBUG`), SearXNG base URL / engines /
categories, scraper concurrency and timeouts, reranker and retrieval knobs
(`TOP_N_URLS`, `TOP_N_CHUNKS`, `VECTOR_CANDIDATES_K`), and bootstrap defaults
(`OLLAMA_BASE_URL`, `DEFAULT_CHAT_MODEL_NAME`, `DEFAULT_EMBEDDING_MODEL_NAME`).
The app refuses to start without a valid `MASTER_KEY`.

## For Developers

### Local Development Environment

1.  Install the [`uv`](https://docs.astral.sh/uv/) package manager:
    `curl -LsSf https://astral.sh/uv/install.sh | sh`.
2.  Install dependencies: `uv sync --all-extras` (Python 3.14).
3.  Start the supporting services (Postgres + SearXNG):
    `podman-compose up -d db searXNG`.
4.  Run the application: `uv run main.py` (API on `http://localhost:3000`).

Roles (`user`/`pro`/`admin`) are seeded at startup; create the first user via
`POST /user/` and log in via `POST /auth/login` (JWT bearer). Role-gated mode access
lands with the `research`/`agents` modes.

### Tests

Unit tests run fully offline (no Postgres, no network):

```bash
uv run pytest
```

Integration tests (require a live Postgres) are deselected by default:

```bash
uv run pytest -m integration
```

Lint:

```bash
uv run ruff check src tests
```

### Contributing

Guidelines for reporting bugs and suggesting features will be added in the future.
`PLAN.md` tracks phased work; `CRUSH.md` documents repo conventions.

## Tech Stack

*   **Backend:** Python 3.14, FastAPI, LangChain, SQLModel + asyncpg, pgvector
*   **Retrieval:** SearXNG (meta-search), trafilatura (extraction), flashrank
    (cross-encoder reranking), bge-m3 embeddings
*   **Frontend:** planned in Vue; **mobile & desktop:** planned in Flutter
*   **Containerization:** Podman/Docker

## License

This project is licensed under the Apache 2.0 License. See the [LICENCE](LICENCE) file
for details.

## Copyright

Copyright 2026 The AskOwl Contributors
