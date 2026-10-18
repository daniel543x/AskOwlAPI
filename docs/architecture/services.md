# Services

The business logic behind the search process.

## Search pipeline (`src/core/ask/`)

1. **Query rewriting** (`services/llm_steps.rewrite_query`) — LLM tool call
   (`SubmitSearchQuery` contract); fallback to the raw query only on a
   tool-protocol error.
2. **Web search + ranking** (`tools/toolbox.web_search`) — SearXNG query →
   flashrank cross-encoder rerank → top URLs; persona domain allowlist and
   per-call engines applied here.
3. **Fetch + chunk** (`tools/toolbox.fetch_pages`) — concurrent page
   downloads (semaphore-bounded), trafilatura extraction in a thread pool,
   chunking; with an embedder configured, chunks are embedded and stored in
   the pgvector chunk store (duplicates skipped).
4. **Context selection** (`tools/toolbox.get_context`) — hybrid retrieval:
   vector-similarity candidates (top-k) → cross-encoder rerank → top-n
   context; graceful fallback to in-memory ranking.
5. **Answer synthesis** (`services/llm_steps.stream_cited_answer`) — LLM
   tool call (`CitedAnswer` contract) over tool-derived context only; the
   streamed tool-call argument is decoded incrementally (`ToolArgStream`)
   for token-by-token SSE. "No context → say so" is deterministic, without
   an extra LLM call.
6. **History** (`services/search._resolve_conversation/_save_turn`) — load
   or create the caller's conversation; prior turns are injected into the
   answer step; the finished turn is persisted.

## Research pipeline (`src/core/research/`)

1. **Planner** (`services/planner.plan_subquestions`) — `ResearchPlan`
   tool call; raw query as the single sub-question on protocol error.
2. **Per-sub-question gathering** — the shared tools per sub-question
   (search → fetch → context), budget-checked between sub-questions.
3. **Synthesis** (`services/synthesis`) — `ResearchReport` tool call over
   the gathered contexts, streamed like the search answer.

## Agent loop (`services/agent.agent_logic`)

The model drives the shared tools in any order until a `SubmitOutcome`
tool call. A `Budget` (iterations / wall-clock / estimated tokens, chars÷4)
bounds every round; exhaustion degrades to one forced synthesis call over
the gathered context instead of failing. Personas bias the system prompt
(context, keywords) and constrain the toolbox (domains, engines) plus
budget overrides.

## Model registry (`src/modules/llm_provider/`)

Admin-curated catalog for all model kinds (chat / embedding / reranker /
decision): providers with encrypted keys, entries with capabilities and
required roles, per-user preferences, global defaults. Resolution order:
request `model_id` → user preference → default → built-in local default.
Models without the `tools` capability are rejected (422) from tool-calling
steps. Provider health is probed at startup and exposed at
`GET /health/models`.

## Identity & access (`src/modules/identity/`, `src/modules/api_keys/`)

JWT login + per-user API keys (hashed, scoped, rate-limited). Role gating:
`user` → search; `research`/`admin` → research + agent; `admin` → catalog and
user management. Blocked users fail every path. See
[api_keys.md](api_keys.md) and [../api/authentication.md](../api/authentication.md).
