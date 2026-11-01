# CRUSH.md — working instructions for the crush agent

## What this project is

AskOwl is a self-hosted, LLM-powered search engine ("private Perplexity"): FastAPI backend
that takes a question, searches the web (SearXNG), fetches and ranks pages, and synthesizes
a cited answer with an LLM. It must serve interactive clients (SSE streaming) **and** other
backends (JSON). Models may be local (Ollama), on the host/LAN, or in the cloud — the
admin-curated model catalog decides. Core rule: **the LLM is never used without a tool call.**

`PLAN.md` is the source of truth for what to build (phases, checklists, definitions of
done). Read the current phase in `PLAN.md` before starting.

## Environment & commands

- Python 3.14, dependencies managed with **uv**: `uv sync --all-extras`.
- Run locally: needs Postgres + SearXNG → `podman-compose up -d db searXNG`, then
  `uv run main.py` (API on :3000).
- Local dev login (admin): `admin@askowl.dev` / `Ask0wlAdmin!2026`.
- Python LSP: use LSP tooling (definitions, references, rename, symbol-aware replaces)
  for navigating and editing Python code, not plain text search.
- Tests: `uv run pytest` — must pass **offline** (no Postgres, no network). DB-dependent
  tests are marked `integration` and skipped by default.
- Lint: `uv run ruff check src tests`.
- No code changes are "done" until unit tests and ruff pass.

## Repo map

- `src/app.py` — FastAPI app + lifespan init. `src/init.py` — bootstrap (db, roles, defaults).
- `src/core/ask/` — ask routes + pipeline (`services/search.py`, `services/chunking.py`).
- `src/modules/llm_provider/` — model registry for all model kinds (chat / embedding /
  reranker / decision): catalog tables, resolution (request → preference → default →
  built-in), provider health probing, admin + user routes.
- `src/modules/embedding/` — `IEmbedder` interface + Ollama implementation (Phase 3 wires
  it into retrieval).
- `src/modules/search_engine/`, `src/modules/scrapy/`, `src/modules/owl_reranker/` —
  pluggable search, scraping, ranking.
- `src/modules/identity/` — JWT auth, users, roles (user/pro/admin).
- `src/modules/history/` — placeholder, PLAN Phase 6.
- `src/tools/` — settings (pydantic-settings), db, crypto. All config goes through
  `settings.py`; never hardcode URLs, secrets, or model names.

## Conventions

- Module pattern: `model.py` (SQLModel tables) / `repository.py` (DB access) /
  `factory.py` (construction + FastAPI `Depends`) / `router.py` (HTTP) /
  `providers/` (interface in `base.py` + implementations).
- Everything async end-to-end: no sync blocking calls (model inference, ranking, chunking,
  `trafilatura.fetch_url`) inside async functions — use `asyncio.to_thread` or async HTTP.
  Heavy models load once at app lifespan, never per request.
- English only: code, comments, commit messages, docs.
- Use the `logging` module (never `print`); debug behavior is gated by `LOG_LEVEL`/`DEBUG`
  settings. Never log API keys or decrypted secrets.
- Errors inside SSE streams are `error` events, never raised `HTTPException` after streaming
  started.
- Every LLM interaction goes through native tool calling (`bind_tools`); models lacking tool
  support are rejected via catalog `capabilities` with a clear error. Never reintroduce
  prompt-parsed JSON as the LLM interface.
- One PLAN phase = one branch = one PR. Tick the boxes in `PLAN.md` when a task is done and
  keep the "Open decisions" table current — do not silently decide open items.
- Versioning: `version = 0.<PR-number>.0` in `pyproject.toml` (PR 40 → `0.40.0`). GitHub
  assigns the number only when the PR opens, so the flow is: push the branch, `gh pr create`,
  read N from the URL, then commit `Set version 0.N.0 (version tracks the PR number)` and
  push again before merge. Never merge with a stale version.
- New DB tables: until Alembic lands (PLAN Phase 6), extend the `create_all` bootstrap in
  `src/init.py` and note the schema change in the PR description.

## Guardrails

- Never commit `.env`, real keys, or model weights. Secrets come from env at startup; a
  missing `MASTER_KEY` must abort startup.
- **API keys for model providers are stored encrypted in the database** (via
  `src/tools/crypto.py`, keyed by `MASTER_KEY`) — this is a design requirement, not a
  preference. Plaintext keys in DB columns, migrations, env files, or logs are a violation.
  Keys are decrypted only in memory when constructing a provider client and are never
  returned by API routes or logged.
- Do not add dependencies without checking they support Python 3.14 and justify them in the
  PR. The app must run with **no** model provider reachable (degrade with clear errors,
  never crash-loop).
- Don't refactor beyond the current PLAN phase's scope; report anything broken you find in
  `PLAN.md` §1 audit terms instead of fixing opportunistically.
