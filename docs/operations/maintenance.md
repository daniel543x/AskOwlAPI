# Maintenance

## Migrations

Schema changes ship as Alembic revisions; startup applies them
automatically (pre-Alembic databases are adopted by stamping). Manual
operations:

```bash
uv run alembic upgrade head      # apply
uv run alembic downgrade -1       # rollback one revision
uv run alembic history            # inspect
```

Validate both directions on a scratch database before shipping a
revision (CI does this via `tests/modules/test_migrations.py`).

## Backup

The state that matters is the Postgres volume (`postgres_data`):

```bash
podman-compose exec db pg_dump -U postgres database > backup_$(date +%F).sql
```

Restore into a fresh volume with `psql` before starting the API
(migrations are idempotent on an already-migrated schema).

## Logs

Structured logging to stdout (`LOG_LEVEL`; `DEBUG=true` adds per-stage
tracing with durations and tool-call payloads — API keys redacted).

```bash
podman-compose logs -f api
```

## Provider health

`GET /health/models` reports the last probe status of every configured
provider; the admin model routes (`/admin/models/...`) list and manage the
catalog. A provider going down degrades to clear errors; it never
crash-loops the app.

## Common repairs

| Symptom | Fix |
|---|---|
| Startup refuses: invalid `MASTER_KEY` | regenerate a Fernet key (see `.env.template`) |
| Provider key rejected after re-keying | provider keys are encrypted with the old `MASTER_KEY`; re-enter them via the admin API |
| API consumers blocked (429) | per-key limit: `API_KEY_RATE_LIMIT_PER_MINUTE`; check `last_used_at` in the key listing |
| User locked out | admin: `PATCH /admin/users/{id}/status` (blocked users fail login, JWT, and API keys) |
