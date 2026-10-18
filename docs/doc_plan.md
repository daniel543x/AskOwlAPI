```text
docs/
├── index.md                 # Main entry point: a short "readme" for the documentation.
├── doc_plan.md              # This file: the intended structure of docs/.
├── api/                     # API Documentation (Interface)
│   ├── authentication.md    # JWT, API keys, roles, per-route auth matrix
│   ├── users.md             # users, roles, personas, api keys, history
│   ├── models.md            # model catalog: providers, entries, defaults
│   ├── ask_search_sse.md    # `search` mode SSE protocol (v2)
│   ├── ask_research_sse.md  # `research` / `agent` mode SSE extensions
│   └── errors.md            # error dictionary, HTTP codes, SSE error events
├── architecture/            # Technical Documentation (Internal)
│   ├── overview.md          # diagrams, general overview
│   ├── database.md          # database schema, migrations
│   ├── services.md          # description of business logic, services
│   ├── api_keys.md          # API keys design (open decision #3)
│   └── decision_layer.md    # Jev/Laya pre-scorers (PLAN Phase 7)
└── operations/              # Formerly "administration" (Maintenance)
    ├── installation.md      # how to set up dev/prod environments
    ├── configuration.md     # environment variables, .env
    └── maintenance.md       # backup, logs, repair scripts
```
