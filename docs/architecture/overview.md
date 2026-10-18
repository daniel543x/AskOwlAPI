# Architecture Overview

```
src/
├── app.py                  # FastAPI app + lifespan (logging, migrations,
│                           #   seeding, ranker preload)
├── init.py                 # bootstrap: Alembic upgrade, roles, catalog probing
├── core/                   # pipelines (per mode)
│   ├── ask/                # `search` mode: rewrite → search → fetch →
│   │   │                   #   hybrid context → cited answer (SSE + JSON),
│   │   │                   #   conversation history wiring, eval harness
│   │   └── services/
│   ├── research/           # `research` + `agent` modes: planner, budgets,
│   │   └── services/       #   agent loop, persona-scoped agents, reports
│   └── tools/              # shared pipeline machinery: AskToolbox
│                           #   (web_search/fetch_pages/get_context), chunking,
│                           #   ToolArgStream, ToolProtocolError
└── modules/                # pluggable capabilities (model/repository/factory/
                            #   router pattern per module)
    ├── identity/           # JWT auth, users, roles (user/research/admin)
    ├── api_keys/           # X-API-Key auth, scopes, rate limiting
    ├── history/            # conversations/messages per user
    ├── personas/           # user-owned agent profiles
    ├── llm_provider/       # model catalog (chat/embedding/reranker), resolution,
    │                       #   health probing, encrypted provider keys
    ├── embedding/          # IEmbedder + Ollama/OpenAI-compatible providers
    ├── chunk_store/        # pgvector chunk repository
    ├── search_engine/      # SearXNG (+ other) search providers
    ├── scrapy/             # page fetching/extraction (trafilatura)
    └── owl_reranker/       # cross-encoder reranking (flashrank)
```

## Core rules

1. **The LLM is never used without a tool call.** Every reasoning step
   (query rewriting, planning, synthesis, agent outcomes) goes through a
   native tool call (`bind_tools`); free-form responses are contract
   violations and become errors, never parsed text.
2. Everything async end-to-end; heavy models load once at app lifespan;
   CPU work (ranking, chunking) runs in threads; page fetches are
   concurrent.
3. Secrets: provider API keys are Fernet-encrypted in the DB (keyed by
   `MASTER_KEY`); user API keys are SHA-256 hashed; plaintext never at
   rest, never logged, never returned.

## Pipelines

- **search** (`GET /ask/search`): query rewrite (tool call) → SearXNG web
  search → concurrent scrape → chunking (+ pgvector embedding) → hybrid
  context (vector candidates → cross-encoder rerank) → cited answer
  (tool call), streamed as SSE or one JSON document; turns persist to
  conversation history.
- **research** (`GET /ask/research`): planner decomposes into
  sub-questions (tool call) → per-sub-question gathering with the shared
  tools → cited report (tool call); bounded by iteration/time/token
  budgets.
- **agent** (`GET /ask/agent`): the model drives the shared tools in any
  order, terminating via a `SubmitOutcome` tool call; persona-scoped via
  `persona_id`; bounded by the same budgets with a forced-synthesis
  degrade path.
