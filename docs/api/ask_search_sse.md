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

| Event     | Payload                                             | Emitted                                                  |
|-----------|-----------------------------------------------------|----------------------------------------------------------|
| `status`  | `{"step": "..."}`                                    | Progress updates between stages.                          |
| `query`   | `{"query": "...", "tool_call": true}`                | After query rewriting. `tool_call: false` means the rewrite step fell back to the raw user query (tool-protocol error path). |
| `sources` | `{"sources": [{"url", "title", "score"}, ...]}`      | After web search + ranking (top-N URLs).                  |
| `context` | `{"context": "...", "chunks": N}`                    | After fetch + chunk ranking; the exact context given to the answer step. |
| `message` | `{"text": "..."}`                                    | One per streamed answer delta; concatenate for the full answer. |
| `done`    | `{"status": "completed"}` or `{"status": "no_context"}` | Terminal success. `no_context`: nothing could be extracted; a deterministic notice was emitted as the only `message`. |
| `error`   | `{"error": "..."}`                                   | Terminal failure (no `done` follows).                     |
| `debug`   | any                                                  | Only when `DEBUG=true`. Stage tracing details; not a stable contract. |

## Typical stream (v2)

```
event: status
id: 1
data: {"step": "Building query for search engine..."}

event: query
id: 2
data: {"query": "rust programming language", "tool_call": true}

event: status
id: 3
data: {"step": "Searching URLs..."}

event: sources
id: 4
data: {"sources": [{"url": "https://...", "title": "...", "score": 0.93}]}

event: status
id: 5
data: {"step": "Downloading data..."}

event: status
id: 6
data: {"step": "Selecting context..."}

event: context
id: 7
data: {"context": "...", "chunks": 12}

event: status
id: 8
data: {"step": "Generating answer..."}

event: message
id: 9
data: {"text": "Rust is "}

...

event: done
id: 27
data: {"status": "completed"}
```

## JSON mode

`GET /ask/search?stream=false` returns one document with the same data:

```json
{
  "query": "what is rust?",
  "optimized_query": "rust programming language",
  "sources": [{"url": "...", "title": "...", "score": 0.93}],
  "context": "...",
  "answer": "Rust is ... [Source](https://...)"
}
```

Errors that occur before streaming starts are HTTP errors (500 pipeline
failure, 502 answer-contract violation). With SSE, the same failures arrive as
`error` events.

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
