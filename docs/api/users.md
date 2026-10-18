# Users & Personal Data

All timestamps are ISO 8601 UTC strings. All ids are UUIDs.

## User management

- `POST /user` — register. 201 `UserRead`, 400 on duplicate email/nickname.
- `GET /user/{id}` — read a user (`UserRead`).
- `GET /user/nn/{nickname}` — read a user by nickname (`UserRead`).
- `PATCH /user/{id}` — update; 404 unknown id, 400 on duplicate fields.
- `DELETE /user/{id}` — delete; 204, 404 unknown id.

> **Known gap**: the four routes above currently have **no authentication
> dependency wired** (see the auth matrix in
> [authentication.md](authentication.md) and PLAN §1 audit). Do not expose
> them publicly until fixed.

### Schemas

`UserCreate` (request body for `POST /user`):

```json
{
  "email": "user@example.com",
  "nickname": "owlfan",
  "password": "s3cret!"
}
```

`UserUpdate` (request body for `PATCH /user/{id}`) — all fields optional:

```json
{ "email": null, "nickname": null, "password": null }
```

`UserRead` (response):

```json
{
  "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "email": "user@example.com",
  "nickname": "owlfan",
  "is_active": true,
  "created_at": "2026-10-17T12:00:00Z",
  "updated_at": "2026-10-17T12:00:00Z"
}
```

`GET /auth/users/me` (any auth) returns the caller's `UserRead`.

Registration assigns the default `user` role; elevated roles (`research`,
`admin`) are granted by an admin via the role endpoints below.

## Roles (admin only)

- `POST /roles/` — create a role (`{"name": "..."}`, 409 on duplicate name).
- `GET /roles/` — list roles (`[{"id", "name"}]`).
- `POST /roles/users/{user_id}/assign?role_name=research` — grant a role.
  `role_name` is a **query parameter**, not body. 404 when the user or role
  is unknown, or the user already has the role.
- `DELETE /roles/users/{user_id}/remove?role_name=research` — revoke a role.

## Admin: user management

- `GET /admin/users?limit=50&offset=0` — list users ordered by `created_at`.
  `limit` 1–200 (default 50), `offset` ≥ 0 (default 0). Returns `UserRead[]`
  (not paginated metadata; clients page by offset).
- `PATCH /admin/users/{id}/status` — block/unblock (`{"is_active": false}`).
  422 when the body has nothing to update, 409 when an admin blocks their own
  account, 404 unknown id. Blocked users fail login (401), fail JWT
  validation (403), and their API keys stop resolving (401).

## Personas (per user)

Own personas only; another user's persona is indistinguishable from a missing
one (404). Use one with `GET /ask/agent?persona_id=...`.

- `POST /personas` — create (201 `PersonaRead`).
- `GET /personas` — list own personas (`PersonaRead[]`, newest first).
- `GET /personas/{persona_id}` — read one.
- `PATCH /personas/{persona_id}` — partial update.
- `DELETE /personas/{persona_id}` — delete (204).

### Schema

`PersonaCreate` / `PersonaUpdate` (update: all fields optional):

```json
{
  "name": "rust-expert",
  "system_context": "Focus on the Rust ecosystem.",
  "allowed_domains": ["rust-lang.org", "docs.rs"],
  "allowed_engines": ["google"],
  "allowed_keywords": ["rust"],
  "default_parameters": { "max_iterations": 6 }
}
```

| Field | Type | Constraints / meaning |
|---|---|---|
| `name` | string | ≤ 100 chars, required on create. |
| `system_context` | string | Free-form context injected into the agent's system prompt. |
| `allowed_domains` | string[] | Host-suffix allowlist for search results; `[]` = unconstrained. |
| `allowed_engines` | string[] | Restricts the SearXNG engines used. |
| `allowed_keywords` | string[] | Keyword biases for query rewriting. |
| `default_parameters` | object | Runtime defaults (e.g. `max_iterations`); can only **lower** budgets, never raise them. Unknown keys are ignored. |

`PersonaRead` adds `id`, `created_at`, `updated_at`.

## API keys (per user)

`POST/GET/DELETE /api-keys` — own keys only. The raw key appears exactly
once at creation. Scopes: `{search, research, agent}`.
Design: [../architecture/api_keys.md](../architecture/api_keys.md).

### Schema

`ApiKeyCreate`:

```json
{ "name": "ci-bot", "scopes": ["search"], "expires_at": "2027-01-01T00:00:00Z" }
```

(`scopes` defaults to `["search"]`; unknown scopes are a 422. `expires_at`
optional, ISO 8601.)

`ApiKeyRead` (list response):

```json
{
  "id": "uuid",
  "name": "ci-bot",
  "key_prefix": "ak-abc12",
  "scopes": ["search"],
  "is_active": true,
  "expires_at": "2027-01-01T00:00:00Z",
  "last_used_at": null,
  "created_at": "2026-10-17T12:00:00Z"
}
```

Creation (`POST`) returns `ApiKeyCreated` = `ApiKeyRead` plus `raw_key`
(`"ak-..."`, exactly 46 chars starting with `ak-`). The key **cannot be
recovered later** — only its SHA-256 hash is stored. `DELETE /api-keys/{id}`
revokes (204).

## Conversation history (per user)

- `GET /history` — own conversations, newest first. **Unpaginated.**
- `GET /history/{id}` — conversation with its messages.
- `DELETE /history/{id}` — delete conversation and messages (204).

### Schema

`ConversationRead`: `{id, title, created_at, updated_at}` (`title` ≤ 200
chars, taken from the first query). `GET /history/{id}` returns
`ConversationWithMessages`, adding:

```json
{
  "messages": [
    { "id": "uuid", "role": "user", "content": "...", "created_at": "..." },
    { "id": "uuid", "role": "assistant", "content": "...", "created_at": "..." }
  ]
}
```

`role` is `"user"` or `"assistant"`. Turns are appended automatically by
`POST /ask/search` (pass `conversation_id` to continue one; omit to create a
new conversation — its id is returned in the JSON response and the SSE
`conversation` event). Unknown `conversation_id`: 404 (JSON mode) or a
terminal `error` event (SSE).
