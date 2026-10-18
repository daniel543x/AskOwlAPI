# Ask Search API — SSE Protocol

Applies to `POST /ask/search` (body: `{"query", "stream", "conversation_id"}`).
The JSON mode (`stream=false`) returns the same data as one document (see
below). POST because a turn persists conversation history; model-selection
params (`model_id`, `embedder_id`, `reranker_id`) stay on the query string.

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
| `sources`      | `{"sources": [{"url": str, "title": str, "score": float (0..1, higher = more relevant)}, ...]}` | After web search + ranking (top-N URLs). |
| `context`      | `{"context": "...", "chunks": int, "vector_retrieval": bool}` | After fetch + chunk ranking; the exact context given to the answer step. `vector_retrieval: true` when candidates came from the pgvector store (hybrid retrieval) instead of the reranker alone. |
| `message`      | `{"text": "..."}`                                    | One per streamed answer delta; concatenate for the full answer. |
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
    "sources": [{"url": "...", "title": "...", "score": 0.93}],
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
