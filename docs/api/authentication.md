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
  to use it. Elevated modes additionally require the `pro`/`admin` role.
- Rate limited per key: `API_KEY_RATE_LIMIT_PER_MINUTE` (default 60/min,
  429 + `Retry-After` when exceeded).
- Keys whose owner is blocked stop working immediately.

## Role model

| Role | Modes | Management |
|---|---|---|
| `user` | `search` | own personas / api-keys / history |
| `pro` | `+ research`, `agent` | — |
| `admin` | everything | models catalog, users, roles |

Admin user management: `GET /admin/users`, `PATCH /admin/users/{id}/status`
(`{"is_active": false}` blocks; admins cannot block themselves).
