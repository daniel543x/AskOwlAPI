# AskOwl Improvement Plan

Status: draft v1 (2026-10-17). Source of truth for phased work. `CRUSH.md` tells the dev agent
(crush) how to work in this repo; this file tells it what to build. (Replaces the retired
`ROADMAP.md`; still-valuable roadmap ideas were folded into Phase 6 and §4.)

---

## 0. Product decisions (locked 2026-10-17)

1. **Model selection is admin-curated.** The admin manages the catalog of providers and models
   (local Ollama, cloud, host/LAN — any mix). Users may only choose from catalog entries the
   admin made available to their role. Defaults (per function) apply when the user does not
   specify anything.
2. **Pre-scorers are "typed decision" models, not cross-encoders.**
   - **Jev** (https://typesafe.ai) — closed, API-only, early access. Returns typed decisions
     (choice / score / yes-no) with calibrated probabilities.
   - **Laya** (https://huggingface.co/convaiinnovations/laya) — open weights, Apache-2.0,
     ModernBERT-large based decision model (421M, 512 tokens; multilingual variant 322M/8k).
     Python SDK (`laya`), ONNX extra (`laya[onnx]`), and `laya[serve]` HTTP server exposing a
     **Jev-compatible API**.
   - One Jev-compatible HTTP client in AskOwl serves both (Laya self-hosted sidecar/host
     service, or Jev cloud endpoint). In-process Laya SDK is a possible later optimization.
   - Laya caveats to respect: base checkpoints are near chance zero-shot (accuracy comes from
     fine-tuning) and outputs are over-confident until temperature scaling. Never trust them
     in the hot path without passing the eval harness (Phase 3).
   - The existing flashrank cross-encoder (`ms-marco-MiniLM-L-12-v2`) remains the relevance
     scorer family; Jev/Laya form a second family for filtering/routing/verification decisions.
3. **Deployment topology.** The application runs in a container. Models may be cloud, on the
   host, on the LAN, or hybrid — therefore every model endpoint (base URL, API key) is catalog
   data, never a hardcoded `localhost`. Ollama is optional, not assumed. Local-first defaults
   are still desirable, but must degrade gracefully (health probes, clear errors).
4. **Ship defaults.** Chat: `qwen3.5` via local Ollama (as used during development).
   **Open decision:** default embedding model — proposal: `bge-m3` or `nomic-embed-text`
   (Ollama), 1024/768 dims. Must be decided before the Phase 3 pgvector schema is written;
   store dimensions per catalog entry so multiple embedders can coexist.
5. **Audience & auth.** AskOwl serves interactive clients (future UI) **and** other backends:
   streaming SSE and non-streaming JSON are both first-class. API-key auth is required but the
   scheme is not finalized — design lands in Phase 6. Roles gate **modes**:
   - `search` — basic search (pipeline). All authenticated users.
   - `research` — advanced multi-step search. Elevated role.
   - `agents` — custom search agents: user-created **personas** specialized in a topic (e.g.
     a programmer persona scoped to a tech stack) so users don't repeat context and the agent
     memory stays clean. Elevated role.
6. **No clients exist yet.** Breaking changes to the SSE event protocol and route semantics
   are allowed until first consumers appear.
7. **No LLM usage without tool calls.** Phase 2 rebuilds every LLM step on native tool
   calling; models without tool support are capability-gated out of steps that need it.
8. **Language & logging.** English everywhere (code, comments, commits, docs). Real `logging`
   with a debug mode (env-gated) that traces each pipeline stage and tool call.

---

## 1. Current-state audit (2026-10-17)

Working: modular FastAPI app; identity (JWT, users, roles user/pro/admin); `llm_provider`
(catalog tables, per-user preference, per-request `model_id`, LangChain `init_chat_model`
factory, encrypted API keys); SearXNG + trafilatura + flashrank pipeline behind
`/ask/search` (SSE + JSON); `create_all` bootstrap seeding Ollama/qwen3.5.

Defects (fix in Phase 0 unless noted):

- `src/core/ask/services/search.py:241` — SSE event emits `da ta:` (corrupted) and lacks the
  `\n\n` terminator; `search.py:245` passes the whole `{context, sources, query}` dict into
  the answer prompt instead of `data["context"]`. Streaming is effectively broken.
- `search.py:66–141` — `sse_search_generator` is dead duplicated pipeline code.
- `search.py:257` — `HTTPException` raised inside the streaming generator after headers are
  sent; kills the stream instead of emitting an SSE `error` event.
- `src/modules/owl_reranker/factory.py:6` — cross-encoder loaded from disk on **every
  request**; ranking and chunking are sync CPU work on the event loop (blocks all concurrent
  requests).
- `src/modules/scrapy/providers/trafilatura.py:20` — `trafilatura.fetch_url` called
  sequentially inside `async def`; blocks the event loop for all page downloads.
- `src/modules/search_engine/providers/searxng.py:10` — base URL hardcoded; factory passes DB
  settings as kwargs the constructor does not accept (crash once settings exist).
- `compose.yml` — image `searxng:latest` does not exist (use `searxng/searxng`), `BASE_URL`
  should be `SEARXNG_BASE_URL`, JSON output format is never enabled in SearXNG settings, db
  service mounts the searxng settings volume, api service commented out, no optional Ollama
  service.
- `src/tools/settings.py:9` — hardcoded fallback secret; must fail fast instead.
- Empty placeholders: `src/modules/llm_provider/router.py`, `src/modules/history/*`,
  `/ask/research` (returns ranked links only), `/ask/deep_research` (returns `None`).
- `print()` used as logging; debug prints in `search_json_adapter`; mixed PL/EN comments.
- Tests: 26 lines (identity only), conftest requires live Postgres. No Alembic, no CI.
- `psycopg2-binary` dependency unused (app uses asyncpg).
- README quickstart broken (`uv pip install -r pyproject.toml` is not valid uv usage).
- Found 2026-10-17 (while registering the OpenRouter provider): `GET /health/models` returns
  500 — `ModelRepository.probe_providers()` (`src/modules/llm_provider/repository.py`) commits,
  which expires the provider ORM instances; the route then reads `provider.id` and the lazy
  refresh raises `MissingGreenlet` (same trap `_set_default_if_missing` already documents).
  Health state is persisted before the crash, so `GET /admin/models/providers` shows it.
  **Fixed 2026-10-17** (phase-3 branch): `probe_providers` now refreshes each provider after
  the commit; regression test in `tests/modules/llm_provider/test_repository.py`
  (integration-marked).
- Found 2026-10-18 (while writing the API docs auth matrix): the user-management routes
  `GET /user/{id}`, `GET /user/nn/{nickname}`, `PATCH /user/{id}`, `DELETE /user/{id}`
  (`src/modules/identity/user/router.py`) have **no authentication dependency wired** — any
  anonymous caller can read, modify (including password), or delete any user by id. The docs
  mark these as a known gap; fix before any public deployment (wire `get_current_user` +
  ownership checks, or fold under `/admin` + self-service `/auth/users/me`).
  **Fixed 2026-10-26** (`fix/user-routes-auth` branch): the four routes now require
  `get_current_user` and enforce self-or-admin visibility — foreign targets return 404
  (not 403) so ids cannot be probed, matching the personas convention. `POST /user`
  stays open as registration. Covered by 4 integration tests (anonymous 401, foreign
  404, self 200, admin 200/204); docs auth matrix + users page updated.
- Found 2026-10-26 (flagged by a task agent building the API client; verified live):
  every commit-then-read auth path returned 500 in production — `POST /auth/refresh` on
  success (`rotate()`'s commit expires the loaded `User`, then reading `user.id` raises
  `MissingGreenlet`), API-key requests (`resolve()` commits `last_used_at`, the caller
  then reads `api_key.id`), admin block and password-change serialization. Root cause
  is systemic, not per-route: production request sessions ran with the default
  `expire_on_commit=True` while the whole test suite (conftest) runs `False`, so the
  class was invisible to every test. **Fixed 2026-10-26** (`fix/session-expiry` branch):
  app sessions are created with `expire_on_commit=False` — matching the test harness
  and the Phase 3 `probe_providers` refresh precedent; regression test pins the session
  configuration (`tests/tools/test_db_session.py`). Verified live: successful refresh,
  API-key request, admin block all 200.

---

## 2. Phases

Each phase = one branch, one PR, boxes ticked here when done.

### Phase 0 — Stabilize (no behavior change)

- [x] Delete dead `sse_search_generator`; keep `search_logic` + adapters as the only pipeline.
- [x] Fix SSE adapter: correct `data:` framing + `\n\n` terminators; pass `data["context"]`
      to the answer chain; emit `error` SSE events instead of raising `HTTPException` mid-stream.
- [x] Logging: configure `logging` (format with timestamps), add `LOG_LEVEL` and `DEBUG` env
      settings; debug mode logs every pipeline stage with durations and LLM tool-call
      payloads (redacting API keys). Replace all `print()`. Standardize comments to English.
- [x] Reranker: load flashrank model once at app lifespan (not per request); run ranking and
      chunking via `asyncio.to_thread`. Move `top_n` constants (10 URLs / 20 chunks) to settings.
- [x] Scraper: fetch pages concurrently with `httpx.AsyncClient` (semaphore ≈ 5, timeout),
      run `trafilatura.extract` in a thread pool. Keep the `IScraper` interface.
- [x] Search provider: make base URL / engines / categories constructor params wired from DB
      settings with env fallback; stop passing unsupported kwargs.
- [x] `compose.yml`: fix searxng image + `SEARXNG_BASE_URL`; mount a repo `data/searxng/`
      settings file enabling `search.formats: [html, json]`; remove the stray db volume mount;
      add commented/profile `ollama` service; keep `db`.
- [x] Settings hygiene: all config through `src/tools/settings.py` (pydantic-settings);
      remove the hardcoded secret fallback (refuse to start without `MASTER_KEY`);
      drop `psycopg2-binary`.
- [x] Tests: unit tests for chunking, ranker (fakes), factories, and the ask pipeline with
      stub search/scrape/LLM providers — no network, no Postgres. Mark DB-dependent tests
      `integration` (skipped by default). Add `ruff` config.
- [x] README: fix quickstart (`uv sync --all-extras`, `uv run main.py`).

**DoD:** fresh `podman-compose up` + local Ollama with one pulled model serves a correct
streaming and JSON answer; `uv run pytest` passes offline; no `print()` in `src/`.

### Phase 1 — Model registry v2 (admin-curated, local-first, all model kinds)

- [x] Generalize the `llm_provider` pattern into one registry for model kinds:
      `chat`, `embedding`, `reranker` (cross-encoder), `decision` (Jev/Laya). Same tables and
      the same resolution order everywhere: request param → user preference (role-checked) →
      global default → built-in local default.
- [x] Catalog entries carry: provider (base URL + encrypted API key — supports cloud, host,
      LAN), model name, default params, `is_active`, `required_role`, `capabilities`
      (e.g. `tools`, `streaming`), and for embeddings the vector dimensions.
- [x] Startup bootstrap: probe configured providers (Ollama `/api/tags`, HTTP ping for
      others); seed catalog only with what actually responds; record health per provider;
      expose `GET /health/models`.
- [x] Fill in the empty `llm_provider/router.py`:
      `GET /models?kind=…` (role-filtered listing), `PUT /models/{id}/preference`,
      admin CRUD for providers/entries/defaults (admin role).
- [x] Embedding provider module: `IEmbedder` interface + Ollama implementation (async,
      batched). No consumer yet — Phase 3 wires it in.
- [x] Reranker registry: `IRanker` implementations registered by type; flashrank entry stays
      the default. Model selection goes through the registry (kills the hardcoded
      `get_ranker`).

**DoD:** an admin can add an OpenRouter provider + model over the API and a user can select
it per request; with only local Ollama running, defaults resolve without manual seeding.

### Phase 2 — Tool-calling pipeline (`search` mode core)

- [x] Promote pipeline internals to first-class tools over the existing modules:
      `web_search(query)`, `fetch_pages(urls)`, `get_context(query, …)` (rerank/select).
- [x] Query rewriting: replace the prompt + `JsonOutputParser` step with a native tool call
      (`bind_tools`); fallback to the raw query stays only as a *tool-protocol* error path,
      never free-form generation.
- [x] Answer synthesis: cited-answer contract enforced via tool call / structured output;
      the answer step receives only tool-derived context ("no context → say so" rule).
- [x] Capability gating: steps requiring tools reject catalog models without the `tools`
      capability with a clear 4xx + explanation.
- [x] SSE protocol v2, documented in `docs/api/`: events `status`, `query`, `sources`,
      `context`, `message`, `done`, `error` (+ `debug` events when debug mode is on), with
      event ids and a changelog; JSON mode returns the same data in one document.
- [x] Integration tests for the tool-calling flow using a scripted fake chat model
      (asserts tools were actually called — enforces the "no LLM without tool call" rule).

**DoD:** `/ask/search` answers exclusively through tool calls (asserted by tests); protocol
documented; both SSE and JSON consumers served.

### Phase 3 — Hybrid retrieval (embeddings + vector store)

- [x] Decide the default embedding model + dims (open decision §0.4); add pgvector (compose
      image `pgvector/pgvector:pg16`, `CREATE EXTENSION vector` in init).
      **Decided 2026-10-17: `bge-m3`, 1024 dims (Ollama).**
- [x] New `chunks` table: content, metadata (url, headers), embedding vector, source page
      reference. Embed at scrape time with the selected embedder.
- [x] Retrieval in `get_context`: vector similarity candidates (top-k) → cross-encoder rerank
      → top-n context. Embedder and reranker selectable per request via the registry
      (`embedder_id` / `reranker_id` query params).
- [x] Eval harness (pytest + golden set of ~20 queries with known good sources): recall@k,
      MRR, end-to-end latency; a small CLI to compare configurations. New scorers must pass
      before becoming defaults.

**DoD:** answers are produced from vector-retrieved + reranked context; eval report exists;
default configuration beats the Phase 2 baseline on the golden set.
*Status: **closed 2026-10-18** (valid run on local Ollama `bge-m3`, both configs on
identical cached search/scrape inputs; report at `eval_reports/eval_report.json`,
gitignored — numbers recorded here). Vector retrieval was exercised on 20/20 queries and
the gate ("vector must not lose to baseline") passed: vector ≥ baseline on every query,
strict gains in mean MRR 0.074 → 0.082 and mean latency 13.2 s → 5.7 s. Recall ties at
0.125 because search-egress poisoning floors 14/20 queries at zero golden URLs fetched
for *both* configs (the rust query returned peptide articles, k8s returned Gmail login
pages) — a strict "beats" is unmeasurable from this host; re-run from a clean egress
after search-layer mitigation (proxy egress / API-keyed engines / result-sanity
heuristics — a scheduled design decision, see the 2026-10-17 diagnosis below). Bugs found
and fixed by the run: `ChunkRepository.vector_candidates` shipped `:query_vector::vector`
to asyncpg as literal SQL (`text()` mis-parses a bind param followed by `::`) and the
aborted transaction cascaded `InFailedSQLTransaction` onto later inserts — now
`CAST(:query_vector AS vector)` plus rollback-on-error, with live-Postgres regression
tests in `tests/modules/chunk_store/test_repository.py` (integration-marked).*

### Phase 4 — Agent mode (`research` mode + agentic loop)

- [x] Agent loop over the shared tools (`web_search`, `fetch_pages`, `get_context`) with
      iteration/token/time budgets; per-step SSE events; models gated on the `tools`
      capability. `pre_score` joins when the decision layer lands (Phase 7).
- [x] `research` mode = planner (sub-questions via tool call) → per-sub-question search
      pipeline → grounded synthesis report with citations. Replaces the `/ask/research` stub.
- [x] Role gating per §0.5 (`search` all users; `research` elevated).

**DoD:** research mode produces a cited report end-to-end with bounded budgets; role gating
tested.
*Status: **closed 2026-10-18**. Shared tools promoted to `src/core/tools/` (toolbox,
chunking, ToolArgStream, ToolProtocolError) so neither pipeline imports the other. New
`src/core/research/`: planner + per-sub-question gathering + cited-report synthesis
(`/ask/research`, SSE + JSON), generic agent loop over the shared tools with a Budget
(iterations / wall-clock / estimated tokens) and a forced-synthesis degrade path
(`/ask/agent`, SSE). Both elevated modes gate on the existing `pro` or `admin` role
(`get_current_elevated_user` in identity; the `pro` role is now seeded at startup —
schema unchanged). Gating verified against live Postgres (401 anonymous / 403 plain
user / 200 pro and admin); planner, agent loop, and adapters covered by 35 offline
tests using the scripted tool model. Budgets and the extended SSE events (`plan`,
budget-aware `done`) documented in `docs/api/ask_research_sse.md` and `.env.template`.
Open decision #4 stays open — the gate reads a central `ELEVATED_ROLES` tuple, so a
role rename is a one-line change.*

### Phase 5 — Personas (custom agents)

- [x] `agents` (personas) table per user: name, system context (topic/tech stack), search
      constraints (allowed domains/engines/keywords), default parameters.
- [x] Persona-scoped `agents` mode: agent loop constrained by the persona (query rewriting
      biased by persona context, search constrained to its scope); select persona per
      request/conversation; role-gated.
- [x] Persona management endpoints (CRUD, own personas only).

**DoD:** a "Rust programmer" persona demonstrably changes the sources and answers for the
same query vs. no persona.
*Status: **closed 2026-10-18**. New `personas` module (`src/modules/personas/`: model /
repository / router, table `persona` — schema change, extends `create_all` bootstrap
until Alembic lands in Phase 6). CRUD is scoped to the owner (`get_for_user` treats
foreign personas as 404); `/ask/agent?persona_id=` loads one of the caller's personas and
constrains the Phase 4 agent loop: system-context/keyword bias in the system prompt,
domain allowlist as a host-suffix filter in `AskToolbox.web_search`, per-call engines via
the extended `ISearchProvider.search(query, engines=...)` (SearXNG honors it), and budget
overrides from `default_parameters` clamped to the settings ceiling. The DoD is asserted
offline (persona changes both the ranked sources and the system prompt for the same
query); CRUD isolation and the persona-wired agent route verified by 6 integration tests
against live Postgres.*

Post-personas follow-up:

- [x] Re-run the Phase 3 golden-set eval from a clean egress (or after search-layer
      mitigation: proxy egress / API-keyed engines — see backlog) so the "vector beats
      baseline" gate can be measured strictly, not just "does not lose" (recall was
      floored at 0.125 by search-egress poisoning on 14/20 queries; details in the Phase 3
      status note).
  *Status: **done 2026-10-18** (branch `eval-rerun-phase3`). Mitigation first: the eval
  gained a result-sanity filter (topical-overlap guard between query and search results;
  the exact Phase 3 poisoning — peptide articles for Rust queries — is now dropped and
  reported per query, 8 unit tests). Re-run on local Ollama `bge-m3`, both configs on
  identical cached inputs, 20/20 queries on genuine vector retrieval (no fallbacks):
  **vector strictly beats baseline — recall 0.100 vs 0.075, MRR 0.063 vs 0.057, mean
  latency 5.9 s vs 12.4 s**; vector also produced context on 3 queries where the
  baseline had none (cross-query chunk accumulation). So the gate is no longer a tie.
  Egress is still not fully clean, but the damage moved layers: after an engine-suspend
  reset, search returned topically-relevant URLs, while page *fetching* is rate-limited
  from this host (403/429 on StackOverflow, Cloudflare, …), capping absolute recall at
  0.100 (16/20 queries still zero). Lifting that cap needs the backlog items: proxy
  egress or API-keyed search backends.*

### Phase 6 — Platform hardening

- [x] History module (fill the empty files): PostgreSQL conversations/messages, per user,
      wired into ask routes; `conversation_id` handling.
- [x] Admin user management: block/unblock users (`is_active`), enforced by auth middleware
      and all API-key checks; admin listing of users.
- [x] API keys for clients/backends: server-generated keys, hash-only storage (reuse
      `CryptoService`), scopes tied to modes/roles, rate limiting; reject valid keys whose
      user is blocked/inactive; track `last_used_at`. Design doc first
      (open decision — scheme not finalized).
- [x] Alembic migrations; retire `create_all` (Phase 3's pgvector schema migrates too);
      validate upgrade **and downgrade** paths on a clean database.
- [x] CI (GitHub Actions): ruff, unit tests (no services), integration tests with
      postgres + searxng services, container build; dependency vulnerability scan
      (e.g. `pip-audit`) and a secrets-exposure check on every PR.
- [x] Fill `docs/` per `docs/doc_plan.md`; README rewrite (modes, model selection, topology
      options) verified by a fresh-clone test; release tagging.
- [x] Release tagging is deferred to the first tagged release (V1-RC); everything else
      above shipped 2026-10-18.

*Status: **closed 2026-10-18** (release tagging noted above). History: `history` module
(`conversation`/`message` tables, owner-scoped CRUD, `GET /history[...]`), wired into
`/ask/search` — `conversation_id` continues a conversation (404 foreign), omitted creates
one (SSE `conversation` event; JSON `conversation_id` field); prior turns inject into the
answer step; turn persistence never kills a finished stream. Admin: `user.is_active`
enforced at login (401), JWT validation (403), and every API-key resolve (401);
`GET /admin/users`, `PATCH /admin/users/{id}/status` (409 self-block guard). API keys:
`ak-`-prefixed server-generated keys, SHA-256 **hash-only** storage (deliberate deviation
from the "reuse CryptoService" note: user keys are never needed in cleartext after
issuance, so irreversibility beats reversibility; design in
`docs/architecture/api_keys.md`), scopes `{search, research, agent}` enforced per mode on
X-API-Key sessions, in-process fixed-window rate limit (`API_KEY_RATE_LIMIT_PER_MINUTE`),
`last_used_at` tracked. Alembic: initial revision covers all 14 tables (pgvector extension
created in-migration); startup runs `upgrade head`, pre-Alembic databases are adopted by
stamping; `create_all` survives only as the pytest harness bootstrap; upgrade/downgrade/
re-upgrade validated on clean scratch DBs (integration tests + live verification). CI:
`.github/workflows/ci.yml` (ruff, offline pytest, integration pytest with a pgvector
service — SearXNG not needed: integration tests stub the search provider, container
build, `uv export` + `pip-audit`, gitleaks with a `.gitleaks.toml` allowlist for the
committed deterministic test key). Docs filled per `doc_plan.md` (api/architecture/
operations + index); Containerfile rewritten (uv-based, migrations baked in — the old one
referenced a nonexistent requirements.txt and a broken CMD); README extended (history,
API keys, migrations, CI). Fresh-clone test executed as command-coherence verification
locally (no second machine available — noted, not silently passed).*

### Phase 7 — Decision layer (Jev / Laya pre-scorers) — deliberately last

- [x] `decision` provider: Jev-compatible HTTP client (choice / score / noul questions →
      typed answers with calibrated probabilities), configured as a catalog provider
      (Laya sidecar / host service / Jev cloud). Optional in-process Laya SDK variant later.
- [x] Apply temperature scaling / calibration wrapper per Laya docs before trusting scores.
- [x] Wire into the pipeline as **pre-scores** (each individually toggleable, off by default
      until eval'd):
      - search-result filtering: noul "is this result relevant to the query?" before scraping
        (saves fetches);
      - chunk filtering: score "rate relevance 1–5" as a second opinion next to the
        cross-encoder;
      - query choice: pick the best rewritten search query among candidates;
      - grounding check: noul "is this claim supported by the context?" as a post-answer
        verification signal (surfaced, not silently blocking).
- [x] Expose decisions as a tool (`pre_score`) so the Phase 4 agent mode can use them too.
- [x] `compose.yml`: optional `laya-serve` profile (`laya[serve]` image/service).

**DoD:** with a Laya sidecar or Jev key configured, each pre-score use case can be enabled
and measured via the Phase 3 eval harness; failures degrade gracefully (skip, never crash).
*Status: **closed 2026-10-18**. New `decision` module: `IDecisionMaker` interface with
typed helpers (noul / score / choice, single + batch) and `JevCompatibleClient` speaking
the documented wire protocol (`/v1/systemone[\/batch]`); one client serves a
`laya[serve]` sidecar, a host service, or Jev cloud (decision #2: sidecar). Resolution via
the catalog (`kind=decision`, provider type `jev`, optional per request
`decision_model_id`); the dependency degrades to None — the whole layer is optional.
Temperature scaling: every probability passes `softmax(logits/T)` with temperatures from
the entry's `default_params.calibration` keyed by `"<type>:<option-count>"` (defaults are
placeholders — fit on own data per Laya docs, mean ECE 0.213→0.081 when fitted; noted in
`docs/architecture/decision_layer.md`). All four pre-scores wired and toggleable in
settings, default OFF: search-result filter (batched noul, never empties the set), chunk
second-opinion (1–5 with min-score floor), query choice (raw vs optimized), grounding
signal (SSE `grounding` event + JSON `grounding_supported`, never blocking). Agent mode
gains a `pre_score` tool when a decision model resolves. Eval CLI measures the retrieval
pre-scores (`--pre-score-search-filter`, `--pre-score-chunk-filter`,
`--decision-id`). compose: `--profile laya` service (pip-installs `laya[serve]`; no
prebuilt image exists). 37 new offline tests (wire protocol, calibration math, degradation
paths, pipeline surfacing, tool gating). Note: this branch also carries the cherry-picked
decisions commit (`a9d51d3`, pushed to the PR #5 branch after that PR was merged — see PR
description).*

### Phase 8 — Auth: refresh tokens (scheduled 2026-10-26, implemented 2026-10-26)

Today `POST /auth/login` returns only a stateless JWT (`ACCESS_TOKEN_EXPIRE_MINUTES`,
6 h as of 2026-10-26) and clients re-send the password on expiry; API keys cover
machines, not interactive sessions. This phase adds a proper refresh flow.

- [x] Design doc first (`docs/architecture/auth.md`): token format, storage, rotation and
      revocation policy, threat model (open decision #5 — do not start before it is decided).
- [x] `refresh_token` table: hash-only storage (SHA-256, same precedent as API keys —
      plaintext never stored or logged), user FK, expiry, rotation parent link, rotated /
      revoked flags, client label + `created_at` + `last_used_at` (per-session rows: each
      login / device mints its own token so one device can be revoked without the others);
      Alembic migration (create_all survives only as the pytest harness bootstrap).
- [x] Issue at `POST /auth/login` next to the access token; the access token stays
      stateless and DB-validated per request (6 h default, decided 2026-10-26 —
      `get_current_user` checks `is_active` on every call, so the long window does not
      delay revocation).
- [x] `POST /auth/refresh`: single-use rotation on every call — a fresh random secret (new
      hash in a new row, parent-linked); the old row is kept marked `rotated` because reuse
      detection depends on the history; reuse of an already-rotated token revokes the whole
      rotation chain (stolen-token detection).
- [x] Revocation paths: explicit logout endpoint, password change, and admin block —
      `is_active` must gate refresh the same way it already gates JWT validation and
      API-key resolution.
- [x] Cleanup: expired / rotated rows are retained (reuse evidence + audit) and purged
      after a retention window by a periodic in-process task on the app lifespan — expired
      tokens presented at use time just get 401, the row stays until the sweep.
- [x] Settings: `REFRESH_TOKEN_EXPIRE_DAYS` (decided: 90), `REFRESH_TOKEN_RETENTION_DAYS`
      (+ any rotation knobs) in `src/tools/settings.py` and `.env.template`; no hardcoded
      values. The access-token default already moved to 360 min.
- [x] Tests: offline unit tests for issue / rotate / reject / chain-revoke logic; DB paths
      (table + route) integration-marked against live Postgres.
- [x] Docs: auth matrix in `docs/api/` and README login section updated; SSE and JSON
      clients unaffected (refresh is a plain JSON round-trip before streaming starts).

**DoD:** an interactive client holds a long-lived refresh token and mints fresh access
tokens without re-sending the password; a rotated/reused token is detected and the chain
revoked; blocked users cannot refresh.
*Status: **closed 2026-10-26** (same day as scheduling; decision #5 approved as opaque
DB-backed). New `RefreshToken` table (migration `c84e1f7a20bd`; migration test's
table set extended) with hash-only storage and per-session rows; `POST /auth/login`
now returns `{access_token, token_type, refresh_token}`, `POST /auth/refresh` rotates
single-use (old row marked rotated, child minted in one commit) and replays revoke the
whole parent-link chain, `POST /auth/logout` is idempotent 204. Revocation wired into
password change (`PATCH /user/{id}`), admin block, and user deletion. Purge sweep runs
at startup and every 6 h on the app lifespan (`REFRESH_TOKEN_RETENTION_DAYS=30`);
session window `REFRESH_TOKEN_EXPIRE_DAYS=90`, sliding per refresh. 8 offline unit
tests (token shape, state classification, chain walk) + 9 integration tests (rotation,
replay-kills-chain, logout, expiry retention, block, password change, purge). Design in
`docs/architecture/auth.md`; API doc auth matrix updated.*

---

## 3. Open decisions

| # | Decision | Owner | Needed by |
|---|----------|-------|-----------|
| 1 | ~~Default embedding model + dims~~ **Decided 2026-10-17: `bge-m3`, 1024 dims (Ollama)** | daniel | Phase 3 ✅ |
| 2 | ~~Laya packaging~~ **Decided 2026-10-18: HTTP sidecar** (`laya[serve]` container or Jev cloud; one Jev-compatible client) | daniel | Phase 7 |
| 3 | ~~API-key scheme (shape, scopes, rotation)~~ **Decided (approved) 2026-10-18**: implemented design in `docs/architecture/api_keys.md` | daniel | closed |
| 4 | ~~Keep or drop the `pro` role name~~ **Decided 2026-10-18: renamed to `research`** (data migration renames seeded rows; assignments preserved via role id) | daniel | closed |
| 5 | ~~Refresh-token format: opaque random token, DB-backed and revocable (API-key precedent) vs stateless signed JWT (no DB hit, but needs a denylist to revoke)~~ **Decided (approved) 2026-10-26: opaque random token, DB-backed** (SHA-256 hash-only, single-use rotation parent chain; access token 6 h, refresh window 90 days) | daniel | closed |

---

## 4. Backlog (unscheduled; carried over from the retired `ROADMAP.md`)

- **Alternate search backends:** Tavily / SerpAPI / Bing API providers for the `search_engine`
  module, configurable per request or per persona. Doubles as the "API-keyed engines"
  mitigation for the search-egress poisoning diagnosed at Phase 3 closure.
- **API-caller tool for agent modes:** let Phase 4/5 agents call external REST APIs with
  user-provided (encrypted) credentials as an additional first-class tool.
- **Soft delete for user data:** hide-but-preserve semantics (conversations, personas) instead
  of hard row deletion.
