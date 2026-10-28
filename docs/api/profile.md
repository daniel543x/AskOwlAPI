# Profile Preferences

Server-side per-user defaults (PLAN Phase 10, PR 10 / decision 12). The
profile is the client's initialization source — ask requests stay
explicit and stateless; the UI reads this once and pre-fills its
controls.

Resolution order everywhere: **request > profile preference > model
default**. Switching the model resets thinking/effort to that model's
defaults: the server clamps on save, and the `resolved` view is what the
UI renders.

## `GET /profile/preferences?model_id=X`

Returns `{stored, resolved, warnings}`:

- `stored` — what the user saved (may reference a model that has since
  disappeared or become unavailable).
- `resolved` — stored values clamped against the effective chat model:
  `model_id` param if given, else the stored `default_model_id`, else
  the user's normal default chain. Unsupported thinking/effort reset to
  `null` (the model's default); an unknown stored model id clears to
  `null`.
- `warnings` — human-readable list of the clamps that were applied.

## `PUT /profile/preferences`

Replaces the whole set (idempotent). Body:

```json
{
  "default_model_id": "uuid | null",
  "default_mode": "search | research | deep_research | persona | null",
  "thinking": "bool | null",
  "effort": "low | medium | high | max | null",
  "source_types": ["web", "news", "academic", "social"]
}
```

Validated against the catalog; unsupported values are clamped (never a
crash, never a 500): unknown `default_model_id` clears with a warning,
thinking/effort clamp to `null` when the model lacks the capability or
level, unknown `source_types` are dropped. Returns the same
`{stored, resolved, warnings}` shape with the normalized payload.

## Ask-time behavior

Ask routes resolve `request > profile > model default`: an explicit
request field always wins and stays strict (422 on impossible
combinations, PR 9 contract); profile values fill only the slots the
request left unset and clamp silently to the resolved model.
`default_model_id` sits between the explicit `model_id` param and the
per-kind model preference; a stale reference (model deleted) degrades
to the next tier.
