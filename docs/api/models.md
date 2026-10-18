# Model Catalog

One admin-curated registry for every model kind: `chat`, `embedding`,
`reranker`, `decision`. Entries live under providers (an endpoint: cloud,
host, LAN, or in-process). Model selection resolves as **request param →
user preference → global default → built-in local default**, with role
checks at every step.

Providers' API keys are stored encrypted (`api_key` on write, never on read —
responses expose only `has_api_key: bool`).

## User routes (any auth)

### `GET /models?kind=chat`

Catalog entries **available to the calling user's roles** (each entry carries
a `required_role`; `user` sees `user`-level entries, `admin` sees all).
Optional `kind` filter (`chat` | `embedding` | `reranker` | `decision`).
Returns `ModelEntry[]`:

```json
[
  {
    "id": "uuid",
    "provider_id": "uuid",
    "kind": "chat",
    "model_name": "qwen3.5",
    "display_name": "Qwen 3.5 (local)",
    "is_active": true,
    "required_role": "user",
    "capabilities": ["tools"],
    "vector_dims": null,
    "provider_name": "ollama-local",
    "provider_type": "ollama"
  }
]
```

| Field | Type | Meaning |
|---|---|---|
| `model_name` | string | Identifier sent to the provider. |
| `display_name` | string | Human label (may be empty). |
| `capabilities` | string[] | e.g. `tools` (required by the ask pipeline), `embedding`. |
| `vector_dims` | int \| null | Embedding entries only (e.g. 1024). |
| `provider_name` / `provider_type` | string | Denormalized from the provider. |

### `GET /models/preferences`

The caller's per-kind defaults: `[{"user_id", "kind", "selected_model_id"}]`.

### `PUT /models/{model_id}/preference`

Sets the caller's default for the entry's kind. Body: none — the target is
the path id. 404 unknown id, 403 when the entry's `required_role` exceeds the
caller's roles. Returns the `PreferenceRead` shown above.

## Health (admin)

### `GET /health/models`

Re-probes every provider (Ollama `/api/tags`, HTTP ping, ...) and returns
fresh `Provider[]` (schema below) with `health_status` ∈
{`up`, `down`, `local`, `unknown`} and `last_probe_at` updated.

## Admin routes

All under `/admin/models`, admin role required.

### Providers

- `POST /providers` — create (201, 409 duplicate name).
- `GET /providers` — list.
- `PATCH /providers/{provider_id}` — partial update (404 unknown).
- `DELETE /providers/{provider_id}` — delete (204, 404 unknown).

`ProviderCreate`:

```json
{
  "name": "openrouter",
  "provider_type": "openai",
  "base_url": "https://openrouter.ai/api/v1",
  "api_key": "sk-or-...",
  "is_active": true
}
```

`ProviderUpdate`: same fields, all optional (`api_key` replaces the stored
one; sending `null` clears it). `ProviderRead`:

```json
{
  "id": "uuid",
  "name": "openrouter",
  "provider_type": "openai",
  "base_url": "https://openrouter.ai/api/v1",
  "is_active": true,
  "health_status": "up",
  "last_probe_at": "2026-10-17T12:00:00Z",
  "has_api_key": true
}
```

### Entries

- `POST /entries` — create (201, 400 invalid payload).
- `GET /entries?kind=` — list (optional kind filter).
- `PATCH /entries/{model_id}` — partial update (400 invalid, 404 unknown).
- `DELETE /entries/{model_id}` — delete (204, 404 unknown).

`ModelEntryCreate` (fields as in the `ModelEntry[]` item above, plus
`provider_id`; `default_params` is an opaque object of provider kwargs).
`ModelEntryUpdate`: all fields optional; the entry's `kind` and `provider_id`
are immutable. Responses are `ModelEntry` (admin sees all entries regardless
of `required_role`).

### Defaults

- `PUT /defaults/{kind}?model_id=uuid` — set the global default for a kind
  (`model_id` is a **query parameter**). 400 when the entry is missing or
  inactive. Returns `{"kind", "model_id", "model_name"}`.
- `GET /defaults` — all defaults:
  `{"defaults": {"chat": "uuid", "reranker": "uuid"}}`.

## Per-request model selection (ask routes)

Ask routes accept optional query parameters that override preferences for
that single call:

| Query param | Applies to | Type |
|---|---|---|
| `model_id` | chat model | UUID |
| `embedder_id` | embedding model | UUID |
| `reranker_id` | reranker model | UUID |
| `decision_model_id` | decision (Jev/Laya) model | UUID |
| `search_provider` | search provider by name | string |

A chat model without the `tools` capability is rejected with 422 before the
pipeline runs; when nothing is resolvable at all the routes answer 503.
