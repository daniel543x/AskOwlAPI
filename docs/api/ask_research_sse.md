# Ask Research / Agent API — SSE Protocol

Applies to `POST /ask/research` (SSE with `stream=true`, default; JSON with
`stream=false`) and `POST /ask/deep_research` (the deep preset: scout +
reflect + notes). Both routes require an elevated role (`research` or
`admin`); see `docs/api/ask_search_sse.md` for the base wire format (SSE
framing, event ids, terminal events) — this document lists the
extensions.

**Protocol v3 (breaking, PR 21):** both routes are POST-only with the
unified body `{query, stream, conversation_id, thinking, effort,
source_types, background, attachment_ids, depth, persona_id,
allow_model_fallback}`; the
`GET /ask/agent` loop is removed (use `/ask/persona` with
`depth=quick`, or `/ask/deep_research`).

## Route parameters (v3: the body)

All ask routes share one body schema (`AskBody`): `{query` (required),
`stream`, `conversation_id`, `thinking`, `effort`, `source_types`,
`background`, `attachment_ids`, `depth`, `persona_id`}`. The research
routes ignore `depth`/`persona_id`/`allow_model_fallback`; `thinking`/`effort`
follow the PR 9 capability contract. Model-selection parameters (`model_id`,
`embedder_id`, `reranker_id`, `decision_model_id`, `search_provider`)
stay on the query string (see [models.md](models.md));
`search_provider` selects a search provider by name.

`source_types` narrows the search slice for the run (PR 11 / decision
13): an array in the body; empty = provider default. Values are
validated against the active provider's declared set
(`GET /search/sources`) — unknown values are a 422.

`persona_id` selects one of the caller's personas (`POST/GET/PATCH/DELETE /personas`):
its system context and keywords bias the agent's queries, its domain allowlist filters
search results (host-suffix match), its engines narrow the search call, and its
`default_parameters` may lower the budgets (never raise them). A persona's
`model` field (a `chat`-kind catalog entry id, validated at write) wins over
every other tier — see the consent flow below.

## Persona model consent (`model_confirm`)

Persona-only (any depth). If the persona's stored model has become unusable
at ask time (deleted from the catalog, deactivated, lost the `tools`
capability, or the owner's role no longer unlocks it), the turn does **not**
silently swap models:

- SSE (`stream=true`, not `background`): the stream opens with one
  `model_confirm` event and closes — no conversation, no turn, no gates.
- JSON (`stream=false`) or `background=true`: HTTP `409` with
  `detail = {"code": "persona_model_unusable", ...same payload...}`.

Payload (raw catalog entry ids; resolve display names via
`GET /models?kind=chat`; `fallback_model` is an advisory hint only —
the server re-resolves at re-submission):

```json
{"persona_model": "<uuid>", "reason": "inactive", "fallback_model": "<uuid>|null"}
```

`reason` ∈ `missing` (absent or wrong kind) | `inactive` | `lacks_tools`
| `no_role`; unknown tokens render a generic "not usable for agent runs"
line. Consent is a re-submission with `allow_model_fallback: true` in the
body, which continues on the standard resolution chain (request →
profile → per-kind preference → global default).

## Events

Reuses the v2 events (`status`, `sources`, `context`, `message`, `done`,
`error`, `debug`) with these extensions:

| Event     | Payload                                              | Emitted                                                  |
|-----------|------------------------------------------------------|----------------------------------------------------------|
| `plan`    | `{"subquestions": [...], "tool_call": bool}`         | Research only. After planning. `tool_call: false` means the planner fell back to the raw query as the single sub-question (tool-protocol error path). |
| `sources` | `{"sources": [...], "subquery": "..."}`              | After each sub-question's web search (`subquery` identifies which). The agent loop emits it without `subquery` after any `web_search` call. |
| `context` | `{"context": "...", "chunks": N, "vector_retrieval": bool, "subquery": "..."}` | After each sub-question's context selection; `subquery` absent in the agent loop. |
| `message` | `{"text": "..."}`                                    | Report deltas (research) / answer deltas (agent). The agent loop emits the terminal `SubmitOutcome` answer as one delta. |
| `model_confirm` | `{"persona_model": "<uuid>", "reason": "missing\|inactive\|lacks_tools\|no_role", "fallback_model": "<uuid>\|null"}` | Persona-only, terminal: the persona's model is unusable and `allow_model_fallback` was not set. The stream carries this event alone; see the consent section above. |
| `done`    | `{"status": "completed" \| "no_context" \| "budget_exhausted", "reason": "..."?}` | `budget_exhausted`: a budget (iterations / time / tokens) stopped the run early; the answer is based on partial context (`reason` explains which budget). |

JSON mode (`/ask/research?stream=false`) returns one document:
`{query, subquestions, gathered: [{subquestion, sources, context}], report}`.
On a model contract violation it returns 502; on pipeline failure 500.

## Budgets

One budget bounds each research/agent run (settings, see `.env.template`):
`AGENT_MAX_ITERATIONS` (agent tool-call rounds), `AGENT_TIME_BUDGET_SECONDS`
(wall clock), `AGENT_TOKEN_BUDGET` (estimated tokens, chars/4, across all
LLM calls). Research additionally caps sub-questions at
`RESEARCH_MAX_SUBQUESTIONS`. Exhausting a budget never aborts a stream: the
run degrades to whatever context was already gathered.

## Typical research stream

```
event: status        -> Planning sub-questions...
event: plan          -> {"subquestions": ["a", "b"], "tool_call": true}
event: status        -> Researching 1/2: a
event: sources       -> {..., "subquery": "a"}
event: context       -> {..., "subquery": "a"}
event: status        -> Researching 2/2: b
...
event: status        -> Writing research report...
event: message       -> report deltas...
event: done          -> {"status": "completed"}
```

## Typical agent stream

```
event: status        -> Agent tool: web_search(...)
event: sources       -> {...}
event: status        -> Agent tool: fetch_pages(...)
event: status        -> Agent tool: get_context(...)
event: message       -> {"text": "full SubmitOutcome answer"}
event: done          -> {"status": "completed"}
```
