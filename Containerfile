# AskOwl API container: uv-managed deps, migrations baked in.
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Dependencies first (layer cache): lockfile only.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Application code, migrations, and the alembic config (startup runs them).
COPY src ./src
COPY migrations ./migrations
COPY alembic.ini main.py ./

# Run as an unprivileged user (needs a writable home for uv's cache).
RUN groupadd --system askowl && useradd --system --create-home --gid askowl askowl \
    && chown -R askowl:askowl /app
USER askowl

EXPOSE 3000

CMD ["uv", "run", "--no-dev", "--frozen", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "3000"]
