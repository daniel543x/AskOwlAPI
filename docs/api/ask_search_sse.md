# Ask Search API — SSE Protocol

Applies to `POST /ask/search` (body: `{"query", "stream", "conversation_id", "max_urls", "query_variations", "cache"}`).
The JSON mode (`stream=false`) returns the same data as one document (see
below). POST because a turn persists conversation history; model-selection
params (`model_id`, `embedder_id`, `reranker_id`) stay on the query string.
`max_urls` (int, optional, `>= 1`) caps the URLs selected for this turn;
omitted = the `TOP_N_URLS` default, clamped server-side to `TOP_N_URLS_MAX`.
`query_variations` (int, optional, `>= 1`, default 1) fans out rule-based
query variants (WU-2) that merge into one ranked pool; clamped server-side
to `QUERY_VARIATIONS_MAX`.
`cache` (`"on"` | `"bypass"`, default `"on"`) — `"bypass"` forces fresh
search + scrape for this turn (evals, admin debugging); a cache hit is
otherwise invisible except latency.
`background` (`bool`, default `false`) — run the turn as a background
job (PR 12): responds `202` with `{job_id, kind, status, queue_position}`;
the answer is never streamed. Jobs run FIFO with per-user fairness
(decision 19); the InBox routes arrive in PR 13.
`attachment_ids` (array of UUIDs, default `[]`) — uploaded files for
this turn (PR 15); must belong to the continued conversation. Files are
uploaded to `POST /conversations/{id}/attachments` and reach retrieval
via context injection and the agent's `search_files` tool.
`thinking` (`bool | null`, default `null`) — reasoning-mode control
(decision 11): `null` = model default; requesting it on a model without
the `thinking` capability is a 422 (`model_lacks_thinking`).
`effort` (`"low"|"medium"|"high"|"max" | null`, default `null`) —
reasoning-effort control (decision 11); requires the `effort`
capability and a supported level (422 `model_lacks_effort` /
`unsupported_effort_level`).
`source_types` (array of strings, default `[]`) — narrows the search
slice for this turn (PR 11 / decision 13). Empty = the provider's own
defaults; validated against the provider's declared set
(`GET /search/sources`) — unknown values are a 422.
`scope` (`"global"` | `"conversation"` | `"request"`, default `"global"`) —
retrieval scope (WU-5): `"global"` (today's behavior) retrieves from the
whole chunk store; `"conversation"` restricts vector candidates to this
conversation (+ legacy unscoped rows); `"request"` restricts retrieval to
this turn's fetched pages.
`recency_bias` (float, `>= 0`, default `0`) — recency boost for chunk
ranking (WU-11): 0 = neutral; publication-fresh chunks (<30 days) get the
largest boost, unknown dates are never penalized; clamped server-side to
`RECENCY_BIAS_MAX`.

Current protocol version: **v2** (breaking; there are no external consumers yet).
Changelog at the bottom.

## Wire format

Standard Server-Sent Events: one frame per event, `Content-Type: text/event-stream`.

```
event: <event name>
id: <sequential integer, 1-based, per stream>
data: <single-line JSON object>

```

Every frame ends with a blank line (`\n\n`). Consumers should treat `done` and
`error` as terminal events. Event `id`s increase monotonically within a stream
and can be used for reassembly/debugging.

## Events

| Event          | Payload                                             | Emitted                                                  |
|----------------|-----------------------------------------------------|----------------------------------------------------------|
| `conversation` | `{"conversation_id": "<uuid>", "created": bool}`    | First event (when history persistence is on): the conversation this turn is stored under. `created: false` when an existing `conversation_id` was continued. |
| `status`       | `{"step": "..."}`                                    | Progress updates between stages.                          |
| `query`        | `{"query": "...", "tool_call": true}`                | After query rewriting. `tool_call: false` means the rewrite step fell back to the raw user query (tool-protocol error path). |
| `sources`      | `{"sources": [{"url": str, "title": str, "score": float (0..1, higher = more relevant), "fusion_score": float or null}, ...]}` | After web search + ranking (top-N URLs, capped by `max_urls` when given). Results are deduped by normalized URL (tracking params and `www.` do not create duplicates), and topically unrelated results are dropped by the sanity filter (`RESULT_SANITY_FILTER`). `fusion_score` is the deterministic engine-agreement/domain-authority/position score used as the cross-encoder tie-breaker. |
| `context`      | `{"context": "...", "chunks": int, "vector_retrieval": bool}` | After fetch + chunk ranking; the exact context given to the answer step. `vector_retrieval: true` when candidates came from the pgvector store (hybrid retrieval) instead of the reranker alone. |
| `message`      | `{"text": "..."}`                                    | One per streamed answer delta; concatenate for the full answer. |
| `thinking`     | `{"text": "..."}`                                    | Reasoning-trace deltas from the synthesis model (v2.8), interleaved before/alongside `message`. Optional — the UI may hide them; concatenate for the full trace. Only models surfacing a reasoning stream produce them. |
| `grounding`    | `{"supported": float (0..1)}`                        | After the last `message`, only when the decision layer (Jev/Laya) is configured and `PRE_SCORE_GROUNDING_CHECK=true`: confidence that the answer is grounded in the context. |
| `done`         | `{"status": "completed"}` or `{"status": "no_context"}` | Terminal success. `no_context`: nothing could be extracted; a deterministic notice was emitted as the only `message`. |
| `error`        | `{"error": "..."}`                                   | Terminal failure (no `done` follows).                     |
| `debug`        | any                                                  | Only when `DEBUG=true`. Stage tracing details; not a stable contract. |

## Typical stream (v2)

```
event: conversation
id: 1
data: {"conversation_id": "3fa8...", "created": true}

event: status
id: 2
data: {"step": "Building query for search engine..."}

event: query
id: 3
data: {"query": "rust programming language", "tool_call": true}

event: status
id: 4
data: {"step": "Searching URLs..."}

event: sources
id: 5
data: {"sources": [{"url": "https://...", "title": "...", "score": 0.93}]}

event: status
id: 6
data: {"step": "Downloading data..."}

event: status
id: 7
data: {"step": "Selecting context..."}

event: context
id: 8
data: {"context": "...", "chunks": 12, "vector_retrieval": false}

event: status
id: 9
data: {"step": "Generating answer..."}

event: message
id: 10
data: {"text": "Rust is "}

...

event: done
id: 28
data: {"status": "completed"}
```

## JSON mode

Same route, `POST /ask/search` with `"stream": false` in the body. The whole
result arrives as one JSON document wrapped in an `answer` object:

```json
{
  "answer": {
    "query": "what is rust?",
    "optimized_query": "rust programming language",
    "sources": [{"url": "...", "title": "...", "score": 0.93, "fusion_score": 0.62}],
    "context": "...",
    "answer": "Rust is ... [Source](https://...)",
    "conversation_id": "3fa8...",
    "grounding_supported": 0.9871
  }
}
```

- `conversation_id` (`string | null`) — the conversation this turn was
  persisted under (`null` only when persistence is unavailable).
- `grounding_supported` (`float | null`) — decision-layer grounding
  confidence, rounded to 4 decimals; `null` when no decision layer is
  configured.

An unknown `conversation_id` is a 404. Errors that occur before the answer
step are HTTP errors (500 pipeline failure, 502 answer-contract violation).
With SSE, the same failures arrive as `error` events.

## Tool-calling contract

The LLM is never used without a tool call:

- Query rewriting must arrive as a `SubmitSearchQuery` tool call. If a model
  answers in prose instead, the raw user query is used (`query.tool_call:
  false`) — model text is never parsed.
- The answer must arrive as a `CitedAnswer` tool call (streamed argument
  deltas become `message` events). Prose answers are a contract violation and
  terminate the stream with an `error` event (HTTP 502 in JSON mode).
- Models without the `tools` capability are rejected before the pipeline runs
  (HTTP 422 with an explanation).

## Changelog

### v2.14 (2026-10-28, PR 16 — additive)

- Added: image attachments (png/jpeg/webp/gif, <= 10 MB each, <= 4 per
  request) via the same upload route, stored as blobs with orphan TTL
  cleanup. Ask-time gate: an image attachment with a model lacking
  `vision` is a coded 422 `model_lacks_vision` — never a silent drop.
  Images reach the synthesis LLM as content blocks next to the
  retrieved context and never the search engine path; history persists
  references, not base64.

### v2.13 (2026-10-28, PR 15 — additive)

- Added: `POST /conversations/{id}/attachments` (multipart; txt, md,
  csv, pdf, docx, html) — files are parsed server-side into
  conversation-scoped chunks (decision 21; no `scope` default flip).
- Added: `attachment_ids` request field on ask routes — validated
  against the conversation (search mode) or referenced directly
  (research/agent); retrieval surfaces via deterministic context
  injection (search) and the `search_files` agent tool (agent modes).

### v2.12 (2026-10-28, PR 14 — additive)

- Added: `offers` event on agent-mode streams when the optional,
  feature-flagged gg.deals tool is used:
  `{"offers": [{title, current_best, historical_low, stores}, ...]}` —
  the client renders a banner; prices never appear in answer text.

### v2.11 (2026-10-28, PR 13 — additive)

- Added: Inbox routes — `GET /jobs[?seen=]`, `POST /jobs/{id}/seen`,
  `POST /jobs/{id}/cancel` (queued only; running answers 409), and a
  per-user SSE stream `GET /jobs/events` emitting `job_update` events
  `{job_id, kind, status, queue_position, result_summary, error}`.
- Behavior: startup sweep marks jobs left `running` by a restart as
  failed.

### v2.10 (2026-10-28, PR 12 — additive)

- Added: `background` request parameter on ask routes — busy foreground
  turns can auto-convert into queued jobs (`202 {job_id, ...}`). Every
  LLM/search call passes per-resource gates (`model:<provider>`,
  `search:<config>`); `BUSY_POLICY=reject` surfaces a coded 503
  `provider_busy` body.
- Rate-limit responses from providers are honored (Retry-After) when
  retrying background jobs.

### v2.9 (2026-10-28, PR 11 — additive)

- Added: request field `source_types` (array; default `[]` = provider
  default) on ask routes, and `GET /search/sources` returning the
  provider's declared set (decision 13). Unknown values are a 422.

### v2.8 (2026-10-28, PR 9 — additive)

- Added: request controls `thinking` (`bool | null`) and `effort`
  (`"low"|"medium"|"high"|"max" | null`) on ask routes — neutral fields;
  provider spellings (`reasoning_effort`, `reasoning`, z.ai `thinking`)
  stay inside the server's adapters. Unsupported combinations are 422s
  with a coded body: `{"code": "model_lacks_thinking" |
  "model_lacks_effort" | "unsupported_effort_level", "message": ...}`.
- Added: SSE `thinking` event (`{"text": "..."}`) — reasoning-trace
  passthrough from the synthesis stream, emitted before/alongside
  `message` events; the UI may hide it. JSON mode does not surface it.

### v2.7 (2026-10-26, WU-11 + WU-5 — additive)

- Added: request body fields `scope` (`"global"` default) and
  `recency_bias` (default 0). Prompts now carry the current date, chunks
  carry `title`/`site`/`published_at`/`fetched_at` metadata, and old
  chunks are purged at boot (`CHUNK_TTL_DAYS`).

### v2.4 (2026-10-26, WU-10 + WU-2 — additive)

- Added: request body fields `query_variations` (rule-based fan-out,
  default 1 = today's behavior) and `cache` (`"on"` default; `"bypass"`
  forces fresh search + scrape for the turn).
- Behavior: identical (query, engines) search calls within
  `SEARCH_CACHE_TTL_S` are memoized in-process, and scraped pages are
  cached Postgres-side for `PAGE_CACHE_TTL_S`; both are invisible to the
  protocol except latency. Empty results are never cached.

### v2.3 (2026-10-26, AR-2 / WU-3 + WU-6 — additive)

- Added: `sources` entries carry `fusion_score` (deterministic
  engine-agreement + domain-authority + position score; the cross-encoder
  stays the primary ranking signal).
- Added: request body field `max_urls` (optional; default `TOP_N_URLS`,
  clamped to `TOP_N_URLS_MAX`).
- Added: `debug` events with `stage: "web_search"` carrying
  `dropped_results`, `merged_results`, and `max_urls` when `DEBUG=true`.
- Behavior: search results are deduplicated by normalized URL and
  topically unrelated results are dropped by default
  (`RESULT_SANITY_FILTER=true`; set it to `false` for raw passthrough).

### v2.2 (2026-10-18, documentation correction — no protocol change)

- Fixed: JSON mode is `POST` with `"stream": false` (was documented as GET);
  the response is wrapped in an `answer` object and includes
  `conversation_id` and `grounding_supported`.
- Documented previously missing events: `conversation` (first event) and
  `grounding` (decision layer), and the `context.vector_retrieval` field.

### v2 (2026-10-17, PLAN Phase 2)

- Breaking: `query_generated` renamed to `query`; now includes `tool_call`.
- Breaking: new required `sources` and `context` events (previously only an
  internal `context_ready` stage existed and sources were not streamed).
- Breaking: `warning` event removed; the empty-extraction path now terminates
  with `done {"status": "no_context"}` after a deterministic `message`.
- Added: sequential event `id`s on every frame.
- Added: `debug` events when `DEBUG=true`.
- Added: `done` is always the terminal success event (v1 had no terminator
  on success).
- Answers now stream from the `CitedAnswer` tool-call argument.

### v1 (2026-10, PLAN Phase 0)

Initial stabilized framing: `status`, `query_generated`, `message`, `error`,
`warning`, `done`, correct `data:` framing and `\n\n` terminators. Superseded.
