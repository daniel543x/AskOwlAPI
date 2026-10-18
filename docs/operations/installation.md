# Installation

## Requirements

- [Podman](https://podman.io/) (or Docker) + `podman-compose` (or
  `docker-compose`) — for the supporting services or the full stack.
- [uv](https://docs.astral.sh/uv/) — Python 3.14 dependency management.
- An Ollama instance (or any configured model provider) for LLM features.

## Full stack (containers)

```bash
git clone https://github.com/daniel543x/AskOwlAPI.git
cd AskOwlAPI
cp .env.template .env
# Fill the required values (Postgres creds, MASTER_KEY, SEARXNG_SECRET).

podman-compose up -d                   # db + searXNG
podman-compose --profile api up -d     # + API container (migrations run at startup)
podman-compose --profile ollama up -d # + local Ollama
```

The API listens on `http://localhost:3000/` (interactive docs at `/docs`).
With the `ollama` profile, pull the default models once:

```bash
podman-compose exec ollama ollama pull qwen3.5
podman-compose exec ollama ollama pull bge-m3
```

## Development (bare metal)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --all-extras                       # Python 3.14 deps
podman-compose up -d db searXNG            # supporting services only
uv run main.py                             # API on http://localhost:3000
```

Startup order: the app probes configured providers, runs Alembic
migrations, seeds roles and the catalog bootstrap, and preloads the
reranker. It refuses to start without a valid `MASTER_KEY`.

## Tests

```bash
uv run pytest                  # offline unit tests (no services)
uv run pytest -m integration   # needs a live Postgres (creates a scratch DB)
uv run ruff check src tests    # lint
```
