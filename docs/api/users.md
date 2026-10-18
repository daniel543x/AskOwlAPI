# Users & Personal Data

## User management

- `POST /user` — register (`{email, nickname, password}`) → 201 `UserRead`.
- `GET /user/{id}`, `GET /user/nn/{nickname}` — read a user.
- `PATCH /user/{id}` — update own email/nickname/password.
- `DELETE /user/{id}` — delete.

Registration assigns the default `user` role; elevated roles (`research`,
`admin`) are granted by an admin via `POST /roles/users/{user_id}/assign`.

## Admin: user management

- `GET /admin/users?limit&offset` — list users (admin role).
- `PATCH /admin/users/{id}/status` — block/unblock
  (`{"is_active": false}`). Blocked users fail login (401), fail JWT
  validation (403), and their API keys stop resolving (401). Admins cannot
  block their own account (409).

## Personas (per user)

`POST/GET/PATCH/DELETE /personas` — own personas only; another user's
persona is indistinguishable from a missing one (404). A persona carries a
system context, keyword biases, domain/engine constraints, and default
parameters. Use one with `GET /ask/agent?persona_id=...`.

## API keys (per user)

`POST/GET/DELETE /api-keys` — own keys only. The raw key appears exactly
once at creation. Scopes: `{search, research, agent}`.
Design: [../architecture/api_keys.md](../architecture/api_keys.md).

## Conversation history (per user)

- `GET /history` — own conversations, newest first.
- `GET /history/{id}` — conversation with its messages.
- `DELETE /history/{id}` — delete conversation and messages.

Turns are appended automatically by `POST /ask/search` (pass
`conversation_id` to continue one; omit to create a new conversation — its
id is returned in the JSON response and the SSE `conversation` event).
