# Ask Research / Agent API — SSE Protocol

Applies to `GET /ask/research` (SSE with `stream=true`, default; JSON with
`stream=false`) and `GET /ask/agent` (SSE only). Both routes require an
elevated role (`research` or `admin`); see `docs/api/ask_search_sse.md` for the
base wire format (SSE framing, event ids, terminal events) — this document
only lists the extensions over protocol v2.

## Route parameters

| Route           | Parameters                                    |
|-----------------|-----------------------------------------------|
| `/ask/research` | `query` (required), `stream` (default `true`), `max_iterations` (default `1`, clamped to `RESEARCH_MAX_ROUNDS`), `model_id`, `embedder_id`, `reranker_id`, `search_provider` |
| `/ask/agent`    | `goal` (required), `persona_id` (optional), `model_id`, `embedder_id`, `reranker_id`, `decision_model_id`, `search_provider` |

Model-selection parameters are optional UUIDs (see
[models.md](models.md)); `search_provider` selects a search provider by name.

`persona_id` selects one of the caller's personas (`POST/GET/PATCH/DELETE /personas`):
its system context and keywords bias the agent's queries, its domain allowlist filters
search results (host-suffix match), its engines narrow the search call, and its
`default_parameters` may lower the budgets (never raise them).

## Events

Reuses the v2 events (`status`, `sources`, `context`, `message`, `done`,
`error`, `debug`) with these extensions:

| Event     | Payload                                              | Emitted                                                  |
|-----------|------------------------------------------------------|----------------------------------------------------------|
| `plan`    | `{"subquestions": [...], "tool_call": bool}`         | Research only. After planning. `tool_call: false` means the planner fell back to the raw query as the single sub-question (tool-protocol error path). |
| `reflect` | `{"missing": [...], "follow_ups": [...], "rounds_used": n}` | Research only, when `max_iterations > 1` (WU-7): after each reflection round — what is still missing and the follow-up queries about to be gathered. Follow-up queries already researched are skipped. |
| `sources` | `{"sources": [...], "subquery": "..."}`              | After each sub-question's web search (`subquery` identifies which). The agent loop emits it without `subquery` after any `web_search` call. |
| `context` | `{"context": "...", "chunks": N, "vector_retrieval": bool, "subquery": "..."}` | After each sub-question's context selection; `subquery` absent in the agent loop. |
| `message` | `{"text": "..."}`                                    | Report deltas (research) / answer deltas (agent). The agent loop emits the terminal `SubmitOutcome` answer as one delta. |
| `done`    | `{"status": "completed" \| "no_context" \| "budget_exhausted", "reason": "..."?}` | `budget_exhausted`: a budget (iterations / time / tokens) stopped the run early; the answer is based on partial context (`reason` explains which budget). |

JSON mode (`/ask/research?stream=false`) returns one document:
`{query, subquestions, gathered: [{subquestion, sources, context}], rounds_used, budget_stop_reason, report}`.
On a model contract violation it returns 502; on pipeline failure 500.

## Reflection loop (WU-7 / WU-8)

With `max_iterations > 1`, after the first gathering pass the pipeline
reflects (a `MissingInfo` tool call) on what is still missing and gathers
`follow_up_queries` through the same search pipeline; reflection cycles =
`max_iterations - 1`, clamped to `RESEARCH_MAX_ROUNDS`. With
`RESEARCH_COMPRESS_CONTEXT=true`, each sub-question's context is merged into
a running brief of at most `SUBQUESTION_BRIEF_TOKENS` (a `ContextBrief`
tool call) and synthesis receives briefs instead of raw contexts — the
structural fix for oversized prompts making models silently truncate and
drop tool calls. Both default to the old single-pass, raw-context behavior.
Tool-protocol failures in reflect/compress degrade deterministically
(empty findings / raw concatenation), never abort the stream.

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
event: reflect       -> {"missing": [...], "follow_ups": [...], "rounds_used": 1}   (max_iterations > 1)
event: status        -> Follow-up (round 1): ...
event: sources       -> {..., "subquery": "follow-up query"}
event: context       -> {..., "subquery": "follow-up query"}
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
