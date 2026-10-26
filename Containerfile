# AskOwl API container: uv-managed deps, migrations baked in.
# Multi-stage: the venv and uv wheel cache stay in the builder; the final
# image carries only the runtime files (no cache, no chown duplication).
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Dependencies first (layer cache): lockfile only.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Application code and the alembic config (startup runs them).
COPY src ./src
COPY migrations ./migrations
COPY alembic.ini main.py ./

FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Run as an unprivileged user (needs a writable home for uv's cache).
RUN groupadd --system askowl && useradd --system --create-home --gid askowl askowl

COPY --from=builder --chown=askowl:askowl /app /app

USER askowl

EXPOSE 3000

CMD ["uv", "run", "--no-dev", "--frozen", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "3000"]
