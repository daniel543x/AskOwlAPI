# API Keys — Design (open decision #3, implemented 2026-10-18)

AskOwl serves interactive clients (JWT after `/auth/login`) and other
backends (API keys). This document is the implemented design; final
sign-off is the plan owner's (decision #3).

## Shape

- Server-generated: `ak-` + 43 url-safe random chars (`secrets.token_urlsafe`).
- Shown exactly once, in the `POST /api-keys` response (`raw_key`); never
  again, never logged, never stored in cleartext.
- Stored: SHA-256 hash only (unique, indexed) + a short non-secret
  `key_prefix` (`ak-xxxxx`) so users can tell keys apart in listings.

### Why hashing instead of `CryptoService` encryption

Provider API keys must be *reversible* (the app re-uses them to construct
provider clients), so they are Fernet-encrypted via `MASTER_KEY`. User API
keys are never needed in cleartext after issuance, so they are hashed —
a leaked DB dump does not yield usable keys. Same rule as passwords.

## Storage (`apikey` table)

| Column | Notes |
|---|---|
| `key_hash` | SHA-256, unique index; lookup key for auth |
| `user_id` | owner; all management endpoints are scoped to the owner |
| `scopes` | JSON list; subset of `{search, research, agent}` |
| `is_active` | revocation flag (`DELETE /api-keys/{id}` sets false) |
| `expires_at` | optional; expiry checked at resolve time |
| `last_used_at` | updated on each authenticated use |

## Scopes

Scopes gate **modes**, roles gate **permissions**:

- `X-API-Key` sessions: the key must carry the mode's scope
  (`search` → `/ask/search`, `research` → `/ask/research`, `agent` →
  `/ask/agent`). Elevated modes still additionally require the elevated
  role (`research`/`admin`) for the key's user.
- JWT sessions are not scope-limited (role gates apply as before).

Unknown scopes are rejected at creation (422).

## Rate limiting

Per-key fixed window, in-process (`API_KEY_RATE_LIMIT_PER_MINUTE`,
default 60/minute), 429 with `Retry-After` when exceeded. Single-node by
design; swap for a shared store (Redis) when a multi-node topology
appears. JWT traffic is not limited by this mechanism.

## Rotation

Manual: revoke (`DELETE`) + create a new key. No silent rotation —
issuance is the only moment the raw key exists, so "rotate" is always an
explicit client-visible event. Automatic expiry via `expires_at` is
supported at creation time.

## Auth path

`X-API-Key: <key>` → hash lookup → active + unexpired + owner user active
→ resolve user (rate-limit check) → `request.state.api_key_scopes` set for
`require_scope`. Blocked users invalidate their keys immediately (the
user activity check runs on every key use).
