# Refresh-token sessions (auth design)

Status: implemented (PLAN Phase 8; decision #5 approved 2026-10-26).

## Goal

Interactive clients (future web app, mobile app) hold a long-lived session
without re-sending the password every `ACCESS_TOKEN_EXPIRE_MINUTES` (6 h),
while machines keep using API keys. API keys cover machines, not human
sessions: they are scoped, unscoped per user and impossible to rotate
per device — the wrong credential shape for "log in once, stay logged in".

## Token format (decision #5)

**Opaque random token, DB-backed** — not a stateless JWT:

- Raw token: `rt-` + `secrets.token_urlsafe(32)`; shown to the client
  exactly once. Server stores only SHA-256(raw) — the raw value is never
  needed again (same irreversibility precedent as API keys).
- Postgres is already mandatory (catalog/history/chunks), so a hash lookup
  on refresh costs nothing and buys real revocation. A JWT refresh token
  would need a denylist table anyway to satisfy the revocation requirements
  below — the DB solution with extra steps and none of the benefits.
- The access token stays a stateless JWT; the per-request hot path is
  unchanged. The 6 h access window is safe because `get_current_user`
  re-checks `is_active` on every call, so blocks take effect immediately.

## Lifecycle

One row per issued session (each login / device mints its own token, so a
stolen phone does not log out the desktop):

```
login ──► row A (active, 90 d)
refresh ─► row B (active, 90 d, parent=A)  + A marked rotated
refresh ─► row C (parent=B)                + B marked rotated
replay A ─► theft detected: A, B, C revoked (whole chain)
```

- **Single-use rotation**: `POST /auth/refresh` marks the presented row
  rotated and mints its child in one commit; the response carries a new
  `refresh_token`. Clients must store the latest token only.
- **Reuse detection**: presenting a rotated or revoked row revokes every
  descendant (parent-link walk) before answering 401. A replayed old token
  is evidence the chain leaked; killing the whole chain locks the thief
  out together with the legitimate client (which must re-login).
- **Expiry**: a token unused for `REFRESH_TOKEN_EXPIRE_DAYS` (90) stops
  working (401 at use time). Active sessions renew indefinitely — the
  window slides with every refresh.
- **Retention & purge**: superseded (rotated / expired / revoked) rows are
  kept for `REFRESH_TOKEN_RETENTION_DAYS` (30) as reuse-detection evidence
  and audit history, then deleted by a sweep that runs at startup and
  every 6 h on the app lifespan. The table stays small without losing
  forensic history.

## Revocation paths (all immediate)

| Event | Effect |
|---|---|
| `POST /auth/logout` | Revokes the presented session chain. |
| Password change (`PATCH /user/{id}`) | Revokes all of the user's sessions. |
| Admin block (`PATCH /admin/users/{id}/status`) | Revokes all sessions; `is_active` also gates refresh resolution. |
| User deletion | Refresh rows hard-deleted with the user. |

## Routes

- `POST /auth/login` → `{access_token, token_type, refresh_token}`
- `POST /auth/refresh` `{refresh_token}` → fresh pair (401 on any failure;
  errors are generic so probing learns nothing — unknown, expired and
  revoked all look the same).
- `POST /auth/logout` `{refresh_token}` → 204, idempotent.

Session metadata: rows carry the User-Agent-derived `client_label`,
`created_at`, and `last_used_at` (never returned by routes today; available
for a future "active sessions" admin view).

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `ACCESS_TOKEN_EXPIRE_MINUTES` | 360 | Access-token (JWT) lifetime. |
| `REFRESH_TOKEN_EXPIRE_DAYS` | 90 | Session window, sliding per refresh. |
| `REFRESH_TOKEN_RETENTION_DAYS` | 30 | How long superseded rows survive the purge. |

## Schema

`refresh_token` table (Alembic `c84e1f7a20bd`): `id`, `user_id` (FK),
`token_hash` (unique, indexed), `parent_id` (self-FK, rotation chain),
`client_label`, `created_at`, `expires_at`, `rotated_at`, `revoked_at`,
`last_used_at`.

## Tests

- Offline: token shape/hash, state classification (rotated wins over
  expired), chain-descendant walk — `tests/modules/test_refresh_tokens_unit.py`.
- Live Postgres: full route flow (rotation, replay kills chain, logout,
  expiry, block, password change, purge) —
  `tests/modules/test_refresh_tokens_integration.py` (integration-marked).
