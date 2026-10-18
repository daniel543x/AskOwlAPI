# AskOwl Backend Project Documentation

Welcome to the AskOwl documentation: a self-hosted, LLM-powered search
engine ("private Perplexity") built on FastAPI. AskOwl takes a question,
searches the web (SearXNG), fetches and ranks pages with hybrid retrieval
(pgvector embeddings + cross-encoder reranking), and synthesizes cited
answers with LLMs — every LLM interaction goes through native tool calls.
The documentation is organized by role:

## 1. API (for users and integrators)

Client-side application developers and backend integrators.

_Key documents:_

- [Authentication](api/authentication.md) — JWT login, API keys, roles
- [Users & personal data](api/users.md) — users, personas, api keys, history
- [Ask Search SSE protocol](api/ask_search_sse.md) — `search` mode stream
- [Ask Research / Agent SSE protocol](api/ask_research_sse.md) — elevated modes
- [Errors](api/errors.md) — status codes and SSE error events

## 2. Architecture (for project developers)

Developers working on the AskOwl source code: the "inner workings".

_Key documents:_

- [Overview](architecture/overview.md) — layout, core rules, pipelines
- [Services](architecture/services.md) — the logic behind the search process
- [Database](architecture/database.md) — schema, migrations
- [API keys design](architecture/api_keys.md) — the open decision #3 design

## 3. Operations (for administrators and DevOps)

Deploying, configuring, and maintaining a production instance.

_Key documents:_

- [Installation](operations/installation.md) — dev and container setup
- [Configuration](operations/configuration.md) — environment variables
- [Maintenance](operations/maintenance.md) — migrations, backup, logs, repairs
