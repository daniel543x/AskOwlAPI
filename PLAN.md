# AskOwl Improvement Plan

Status: draft v1 (2026-10-17). Source of truth for phased work. `CRUSH.md` tells the dev agent
(crush) how to work in this repo; this file tells it what to build.

Complements `ROADMAP.md` (rough mapping: Phases 0–1 ≈ Stage 1, Phase 5 ≈ Stage 2 agentic,
Phase 6 personas ≈ Stage 3, Phase 7 ≈ Stages 4–5).

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
   scheme is not finalized — design lands in Phase 7. Roles gate **modes**:
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
*Status: implemented; the measured comparison run (`uv run python -m src.core.ask.eval`)
needs the live stack (Postgres+pgvector, SearXNG, Ollama with `bge-m3` pulled) and is
pending that run before Phase 3 is closed.*

### Phase 4 — Decision layer (Jev / Laya pre-scorers)

- [ ] `decision` provider: Jev-compatible HTTP client (choice / score / noul questions →
      typed answers with calibrated probabilities), configured as a catalog provider
      (Laya sidecar / host service / Jev cloud). Optional in-process Laya SDK variant later.
- [ ] Apply temperature scaling / calibration wrapper per Laya docs before trusting scores.
- [ ] Wire into the pipeline as **pre-scores** (each individually toggleable, off by default
      until eval'd):
      - search-result filtering: noul "is this result relevant to the query?" before scraping
        (saves fetches);
      - chunk filtering: score "rate relevance 1–5" as a second opinion next to the
        cross-encoder;
      - query choice: pick the best rewritten search query among candidates;
      - grounding check: noul "is this claim supported by the context?" as a post-answer
        verification signal (surfaced, not silently blocking).
- [ ] Expose decisions as a tool (`pre_score`) so the Phase 5 agent can use them too.
- [ ] `compose.yml`: optional `laya-serve` profile (`laya[serve]` image/service).

**DoD:** with a Laya sidecar or Jev key configured, each pre-score use case can be enabled
and measured via the Phase 3 eval harness; failures degrade gracefully (skip, never crash).

### Phase 5 — Agent mode (`research` mode + agentic loop)

- [ ] Agent loop over the shared tools (`web_search`, `fetch_pages`, `get_context`,
      `pre_score`) with iteration/token/time budgets; per-step SSE events; models gated on
      the `tools` capability.
- [ ] `research` mode = planner (sub-questions via tool call) → per-sub-question search
      pipeline → grounded synthesis report with citations. Replaces the `/ask/research` stub.
- [ ] Role gating per §0.5 (`search` all users; `research` elevated).

**DoD:** research mode produces a cited report end-to-end with bounded budgets; role gating
tested.

### Phase 6 — Personas (custom agents)

- [ ] `agents` (personas) table per user: name, system context (topic/tech stack), search
      constraints (allowed domains/engines/keywords), default parameters.
- [ ] Persona-scoped `agents` mode: agent loop constrained by the persona (query rewriting
      biased by persona context, search constrained to its scope); select persona per
      request/conversation; role-gated.
- [ ] Persona management endpoints (CRUD, own personas only).

**DoD:** a "Rust programmer" persona demonstrably changes the sources and answers for the
same query vs. no persona.

### Phase 7 — Platform hardening

- [ ] History module (fill the empty files): PostgreSQL conversations/messages, per user,
      wired into ask routes; `conversation_id` handling.
- [ ] API keys for clients/backends: server-generated keys, hash-only storage (reuse
      `CryptoService`), scopes tied to modes/roles, rate limiting. Design doc first
      (open decision — scheme not finalized).
- [ ] Alembic migrations; retire `create_all` (Phase 3's pgvector schema migrates too).
- [ ] CI (GitHub Actions): ruff, unit tests (no services), integration tests with
      postgres + searxng services, container build.
- [ ] Fill `docs/` per `docs/doc_plan.md`; README rewrite (modes, model selection, topology
      options); release tagging.

---

## 3. Open decisions

| # | Decision | Owner | Needed by |
|---|----------|-------|-----------|
| 1 | ~~Default embedding model + dims~~ **Decided 2026-10-17: `bge-m3`, 1024 dims (Ollama)** | daniel | Phase 3 ✅ |
| 2 | Laya packaging: HTTP sidecar (recommended, matches Jev cloud too) vs in-process SDK | daniel | Phase 4 |
| 3 | API-key scheme (shape, scopes, rotation) | daniel | Phase 7 |
| 4 | Keep or drop the `pro` role name once modes are gated | daniel | Phase 5 |
