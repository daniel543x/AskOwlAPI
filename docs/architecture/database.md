# Database

PostgreSQL 16 with the `vector` extension (pgvector) — the compose `db`
service uses the `pgvector/pgvector:pg16` image.

## Schema management

Alembic is the schema's source of truth (`migrations/versions/`). At
startup the app runs `upgrade head` (or, on a pre-Alembic database created
by the old `create_all` bootstrap, stamps the current revision — adoption
is non-destructive).

```bash
uv run alembic upgrade head                        # apply (settings URL)
uv run alembic -x url=postgresql+asyncpg://... upgrade head
uv run alembic downgrade base                      # rollback
uv run alembic revision --autogenerate -m "..."    # new migration
```

`init_db` (create_all) survives only as the pytest harness bootstrap
(tests recreate a scratch database per function).

## Tables

| Table | Purpose |
|---|---|
| `user`, `role`, `userrole` | identity; `user.is_active` blocks auth everywhere |
| `apikey` | per-user API keys: SHA-256 `key_hash`, scopes, `last_used_at` |
| `conversation`, `message` | per-user chat history (user/assistant turns) |
| `persona` | user-owned agent profiles (context, constraints, defaults) |
| `chunk` | scraped content chunks + pgvector `embedding` (hybrid retrieval) |
| `modelprovider`, `modelcatalogentry`, `usermodelpreference`, `defaultsettings` | model catalog for all kinds (chat/embedding/reranker/decision); provider keys Fernet-encrypted |
| `scrapysettings`, `searchproviderconfig` | pluggable search/scrape configuration |

Schema changes: extend the model, autogenerate a revision, validate
upgrade + downgrade paths (`tests/modules/test_migrations.py` does this on
a clean scratch database in CI).
