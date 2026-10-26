# Authentication

AskOwl supports two authentication schemes on the same routes:

| Scheme | Header | Use case |
|---|---|---|
| JWT (login session) | `Authorization: Bearer <token>` | interactive clients |
| API key | `X-API-Key: <key>` | other backends, scripts |

## JWT (login sessions)

1. `POST /auth/login` with `{"username_or_email", "password"}` →
   `{"access_token", "token_type": "bearer", "refresh_token"}` (access
   token expires in `ACCESS_TOKEN_EXPIRE_MINUTES`, default 360 = 6 hours).
2. Send `Authorization: Bearer <access_token>`.
3. When it expires, `POST /auth/refresh` with
   `{"refresh_token": "..."}` → a fresh pair. The refresh token is
   **single-use**: every call retires the presented token and mints a new
   one — always store the latest token from the response.
4. `POST /auth/logout` with `{"refresh_token": "..."}` revokes the session
   (idempotent, always 204).

Login fails (401) for unknown credentials **and for blocked accounts**;
blocked users also fail JWT validation (403) everywhere.

Refresh-token rules (design: docs/architecture/auth.md):

- Hash-only server-side storage; the raw token is shown exactly once at
  login/refresh and never again.
- Replaying an already-rotated or revoked token is treated as theft: the
  whole rotation chain is revoked and every refresh in it returns 401.
- Sessions live `REFRESH_TOKEN_EXPIRE_DAYS` (default 90) from the last
  refresh; password change or admin block revokes all sessions immediately.

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
| `POST /auth/login` | none | Returns access + refresh token. |
| `POST /auth/refresh` | none (refresh token is the credential) | Single-use rotation. |
| `POST /auth/logout` | none (refresh token is the credential) | Idempotent 204. |
| `POST /user` | none | Registration. |
| `GET /user/{id}`, `GET /user/nn/{nickname}`, `PATCH /user/{id}`, `DELETE /user/{id}` | user (self) or admin | Foreign users return 404 so ids cannot be probed. |
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
