# Authentication

AskOwl supports two authentication schemes on the same routes:

| Scheme | Header | Use case |
|---|---|---|
| JWT (login session) | `Authorization: Bearer <token>` | interactive clients |
| API key | `X-API-Key: <key>` | other backends, scripts |

## JWT

1. `POST /auth/login` with `{"username_or_email", "password"}` →
   `{"access_token", "token_type": "bearer"}` (expires in
   `ACCESS_TOKEN_EXPIRE_MINUTES`, default 30).
2. Send `Authorization: Bearer <access_token>`.

Login fails (401) for unknown credentials **and for blocked accounts**;
blocked users also fail JWT validation (403) everywhere.

## API keys (design: docs/architecture/api_keys.md)

- `POST /api-keys` `{"name", "scopes": ["search"], "expires_at"?}` →
   response contains `raw_key` **exactly once**.
- `GET /api-keys` lists keys (hash never returned), `DELETE /api-keys/{id}`
  revokes.
- Scopes: `{search, research, agent}`; the key must carry a mode's scope
  to use it. Elevated modes additionally require the `research`/`admin` role.
- Rate limited per key: `API_KEY_RATE_LIMIT_PER_MINUTE` (default 60/min,
  429 + `Retry-After` when exceeded).
- Keys whose owner is blocked stop working immediately.

## Role model

| Role | Modes | Management |
|---|---|---|
| `user` | `search` | own personas / api-keys / history |
| `research` | `+ research`, `agent` | — |
| `admin` | everything | models catalog, users, roles |

Admin user management: `GET /admin/users`, `PATCH /admin/users/{id}/status`
(`{"is_active": false}` blocks; admins cannot block themselves).

## Auth matrix (per route)

"User" = any authenticated principal (JWT or API key); "elevated" = `research`
or `admin` role; "admin" = `admin` role. API keys additionally need the mode's
scope (`search` / `research` / `agent`); JWT sessions bypass scope checks.

| Route | Auth | Notes |
|---|---|---|
| `GET /healthy` | none | Liveness probe; returns `{"status", "version"}`. |
| `POST /auth/login` | none | Returns the JWT. |
| `POST /user` | none | Registration. |
| `GET /user/{id}`, `GET /user/nn/{nickname}`, `PATCH /user/{id}`, `DELETE /user/{id}` | **none (known gap)** | No auth dependency is wired today; treat as insecure until fixed — tracked in PLAN §1 audit. |
| `GET /auth/users/me` | user | |
| `/personas` (all methods) | user | Own personas only. |
| `/api-keys` (all methods) | user | Own keys only. |
| `/history` (all methods) | user | Own conversations only. |
| `GET /models`, `GET /models/preferences`, `PUT /models/{id}/preference` | user | Role-filtered catalog. |
| `POST /ask/search` | user | + `search` scope on API keys. |
| `GET /ask/research` | elevated | + `research` scope on API keys. |
| `GET /ask/agent` | elevated | + `agent` scope on API keys. |
| `GET /health/models` | admin | |
| `/admin/models/*` | admin | Providers, entries, defaults. |
| `GET /admin/users`, `PATCH /admin/users/{id}/status` | admin | |
| `/roles/*` | admin | |
