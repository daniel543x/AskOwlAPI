# AskOwl Improvement Plan

Status: **draft v2 (2026-10-28)**. Source of truth for phased work. `CRUSH.md` tells the dev
agent (crush) how to work in this repo; this file tells it what to build.

v1 (2026-10-17) drove Phases 0–9: Phases 0–8 are shipped, Phase 9 PR 1 is shipped. This
revision folds in the **webapp-client workstream** decided with the UI client author on
2026-10-28 (client doubts 1–8 plus follow-ups: thinking/effort contract, profile
preferences, attachments/vision). Where v2 supersedes v1 sequencing, §2.1 reconciles it
explicitly — nothing is silently dropped.

---

## 0. Product decisions

### Locked 2026-10-17 (v1, unchanged)

1. **Model selection is admin-curated.** The admin manages the catalog of providers and
   models (local Ollama, cloud, host/LAN — any mix). Users may only choose from catalog
   entries the admin made available to their role. Defaults (per function) apply when the
   user does not specify anything.
2. **Pre-scorers are "typed decision" models, not cross-encoders** (Jev / Laya; one
   Jev-compatible HTTP client; flashrank stays the relevance-scorer family). Phase 7 —
   deliberately last.
3. **Deployment topology:** the app runs in a container; every model endpoint (base URL,
   API key) is catalog data, never a hardcoded `localhost`. Ollama is optional, not
   assumed; local-first defaults must degrade gracefully.
4. **Ship defaults:** chat `qwen3.5` via local Ollama; embeddings `bge-m3`, 1024 dims
   (decided 2026-10-17); dimensions stored per catalog entry.
5. **Audience & auth.** Interactive clients (the webapp) **and** other backends: streaming
   SSE and non-streaming JSON are both first-class. API-key auth + refresh tokens
   (Phase 8). **Roles gate modes — updated 2026-10-28, four modes:**
   - `search` — basic search. All authenticated users.
   - `research` — multi-step research. Elevated role.
   - `deep_research` — autonomous long-loop research (v1 name: `agent`). Elevated role.
   - `persona` — user-created agents; **whole mode is elevated** (depth is a budget
     choice, not a role choice).
6. **No consumers yet — until the webapp ships.** Breaking changes to the SSE protocol and
   route semantics remain allowed; the webapp is the imminent first consumer and its GA
   freezes the contract (see decision 16).
7. **No LLM usage without tool calls.** Every LLM step is a native tool-calling step.
   Consequence: a model usable by AskOwl at all must expose `tools`; **images additionally
   require `vision`** (decision 14).

### Locked 2026-10-28 (v2 — webapp client round)

8. **One agentic core; modes = presets.** The core is one loop (toolbox + conversation
   history) shared by all modes; `search`/`research`/`deep_research`/`persona` are budget +
   feature presets over it (table in §2.3). The deterministic search pipeline
   (rewrite → search → fetch → context → CitedAnswer) is **kept as the degrade path**:
   triggered by a tool-contract violation (1 repair attempt, then degrade) or budget
   exhaustion without material; the SSE stream carries a `degraded` event so the UI can be
   honest about it. Feature flag `AGENTIC_SEARCH` ships default **off** and flips after the
   eval gate (§2.3, PR 17).
9. **Note tool for research-grade modes.** `note_write(task, content, sources_used)` writes
   a `research_note` row; **the DB is the source of truth**. Notes decouple working-set
   growth from the synthesis context and preserve citations across regeneration. Report
   assembly is deterministic: the model calls
   `SubmitReport(sections: [{idx, include, order, lead_in?}], intro?)` and the backend
   streams note-sections + per-note citations + a global Sources section; a small editorial
   call sees only the TOC. Used by `research`, `deep_research`, and `persona` at
   depth ≥ research. Degrade: no notes → today's gathered→generate_report path.
10. **Personas are a separate mode** (`POST /ask/persona`, required `persona_id` — missing
    → 422, foreign → 404) with `depth: quick|research|deep` (default from
    `persona.default_parameters`, clamped via `_bounded()`). UX argument: a card grid with
    explicit opt-in prevents accidentally asking the wrong agent.
11. **Thinking & effort contract (neutral fields, provider adapters).** Request fields are
    **`thinking: bool|null`** and **`effort: "low"|"medium"|"high"|"max"|null`**
    (`null` = model default). The name `reasoning_effort` exists **only** as OpenAI's wire
    parameter inside the adapter — never in our API, docs, or UI. Catalog capabilities:
    `tools`, **`thinking`**, **`effort`** (+ `vision`, decision 14); `effort` capability
    comes with `effort_levels` (e.g. Ollama gpt-oss: low/medium/high; OpenAI: low/medium/
    high; Ollama hybrid models: low/medium/high/max). UI renders controls **only** from
    `GET /models` capabilities — never hardcoded per model family. Requesting a parameter
    without the capability → `422` (precedent: `tools` gating). Transmission via a
    `ReasoningAdapter` registry per `provider_type` (§2.2, PR 9); factory merge order:
    `default_params` (catalog) < adapter output (request wins). **No auto model routing by
    effort** — bigger-model-lower-effort is the user's decision; presets do **not** force
    `thinking` (defaults belong to the profile, decision 12).
12. **Profile preferences (server-side per-user defaults).** Stored: `default_model_id`,
    `default_mode`, `thinking`, `effort`, `source_types`. **Switching the model resets
    thinking/effort to that model's defaults** and the user re-picks (server enforces this
    by clamping on save; the resolved view is what the UI renders). Resolution order:
    **request > profile preference > model default**. `PUT /profile/preferences` validates
    against the catalog and clamps unsupported values, returning the normalized payload +
    `warnings` (degrade, never crash). Requests stay explicit (the API remains stateless;
    the profile is the client's initialization source).
13. **Source types are provider-declared, discovered — not hardcoded.** v1 set:
    `web | news | academic | social`. `ISearchProvider.search(query, engines, categories)`;
    SearXNG map: web→`general`, news→`news`, academic→`science`, social→engines
    (reddit et al. — not a category). `GET /search/sources` returns the types supported by
    the active/selected provider; the UI renders checkboxes dynamically.
14. **Uploads & attachments.** Text files (txt/md/csv via stdlib, pdf via `pypdf`, docx via
    `python-docx`, html via trafilatura) are parsed server-side into
    **conversation-scoped chunks** (requires Phase 9 PR 6 scoping) and surfaced to the agent
    via a `search_files` tool. **Images are accepted only when the model has `vision`:**
    upload via `POST /conversations/{id}/attachments` (multipart) → `attachment_id`; the ask
    request references `attachment_ids`; **at ask time, any image attachment with a
    non-vision model → `422 model_lacks_vision`** (never a silent drop). Iron rule (7) ⇒
    image requests require capabilities ⊇ `{tools, vision}`. Images go to the synthesis LLM
    as content blocks (via LiteLLM), **never to the search engine**. Limits: allowlist
    png/jpeg/webp/gif, ≤ 10 MB each, ≤ 4 images per request; blob storage with TTL cleanup;
    messages persist references, not base64.
15. **InBox = background jobs layer.** `background: true` on research-grade modes →
    `202 + job_id`; `job` table (status, result_summary, seen, error); per-user SSE channel
    with `job_update` events; `GET /jobs`, `POST /jobs/{id}/seen`; InBox is the UI over the
    queue (decision 19): pending list with positions, running progress, completion
    notifications (seen flag). Execution flows through the resource gates + job queue;
    startup sweep marks `running` → `failed` (precedent: lifespan sweeps); notes make runs
    incrementally observable and resumable later.
16. **Versioning: 0.x until the webapp freezes the contract.** Reset `pyproject` from
    `1.0.0rc1` to **0.15.0** in **PR 9 `client/capabilities`** (round-4 confirmation:
    wave-1 merges — PR 2 / PR 6 / PR 11 — ship **without a version change**; the minor
    ladder starts at the reset); **minor bump per merged phase branch** (0.16.0, …);
    **1.0.0 = the webapp ships as first consumer** (contract freeze +
    deprecation policy). SSE protocol changelog discipline starts now (v2.4 additive
    events; v3 = breaking, Phase 11).
17. **Cache: `wu/cache` (Phase 9 PR 2) via `ICache`.** In-memory TTL default; optional
    Redis behind `REDIS_URL` (lazy import, degrade to memory). Retrieval artifacts cached,
    **LLM answers never**; heavy page blobs → Postgres. The rate limiter is refactored onto
    `ICache` in the same PR (behavior unchanged; unlocks multi-node later).
18. **gg.deals is an optional, feature-flagged agent tool** (pattern: conditional attach
    like `pre_score`). **Official API confirmed 2026-10-28: https://gg.deals/api** —
    API key required, current + historical prices, Bundles API also exists. Dedicated
    `src/modules/ggdeals/` module: HTTP client + `game_offers(game_name)` tool returning
    structured offers (object/JSON, not prose). The ask response carries the structured
    payload → SSE `offers` event → the client renders the banner — never inside answer
    text. Exact endpoint shapes, quota and ToS verified at implementation (site is
    Cloudflare-gated; the key appears to be a paid tier). Alternatives if the API
    disappoints: CheapShark, IsThereAnyDeal.
19. **Provider-aware job queue with resource gates** (round 4: cache × inbox × queue).
    Every LLM/search/external-API call passes a **`ResourceGate`** — an async semaphore
    keyed by resource (`model:<provider_id>`, `search:<provider_id>`, `ggdeals`, …) with
    per-provider `max_concurrency` (Ollama ships 1 — GPU-bound, serial; cloud providers
    ship ≥ 2 — parallel). Foreground requests pass gates too (`BUSY_POLICY:
    wait|background|reject` — busy foreground ask can auto-convert into a queued job →
    Inbox); `background: true` jobs queue FIFO with **per-user fairness** (round-robin +
    per-user running cap) so several users never starve each other. External 429s are
    retried with backoff honoring `Retry-After`. `JobQueue` backends: in-process asyncio
    (default, single-node) or Redis (`REDIS_URL`, optional — multi-node, crash-resilient),
    the same lazy pattern as `ICache` (decision 17).
20. **z.ai thinking/effort: native mapping + probe-prune** (round 5, closes open
    decision 11). Verified against the z.ai Deep Thinking docs (2026-10-28): the native
    API supports `thinking: {type: enabled|disabled}` **and** `reasoning_effort`
    (GLM-5.2+; values `max|xhigh|high|medium|low|minimal|none`, default `max`; compat
    aliases `xhigh→max`, `low/medium→high` — whether they apply to the full API or only
    the Coding Plan is exactly what the probe verifies; `minimal/none` skip thinking;
    effort applies only while thinking is enabled; GLM-5.3 can no longer disable
    thinking — effort-only; the response carries `reasoning_content` → SSE `thinking`
    passthrough). Adapter zai: `thinking: false` → `{type: "disabled"}` (GLM ≤ 5.2;
    GLM-5.3+ → `422 model_lacks_thinking`, capability `thinking` absent); `effort` →
    `reasoning_effort` (`low|medium|high|max` — `max` is natively supported). Catalog
    seeds: GLM-5.2+ `effort_levels: [low, medium, high, max]`; GLM-4.6/4.5: `thinking`
    capability only. PR 9 adds a **live probe** (one synthetic call per level):
    aliased/rejected levels are pruned from the cached `effort_levels` (if aliasing is
    confirmed, the list converges to `[high, max]` — the conservative variant, reached
    automatically). Probe rules: scoped to a single catalog entry, **never across
    providers**; precedence **admin override > probe > seed**; a probe may only prune
    confirmed-unsupported levels, never add; on probe error keep the seed (degrade, never
    crash); results cached via `ICache` (PR 2). OpenAI/OpenRouter stay seed/
    metadata-driven — no paid-API probing (OpenAI: documented stable levels; OpenRouter:
    model metadata); deepseek unchanged (two entries, no runtime toggle).
21. **Chunk scope stays targeted — no default flip** (round 5, closes open decision 13).
    Web chunks remain **global** (cache reuse, dedupe, cross-conversation recall are
    features); upload chunks are **conversation-scoped unconditionally** (privacy; PR 15,
    independent of the `scope` default); `scope=conversation` remains an opt-in request
    parameter. Tagging `conversation_id` happens for all chunks in PR 6 anyway, so a
    later flip is a one-default change + eval re-run. Revisit at protocol v3 (PR 21) if
    the instance turns multi-user — then flip to conversation-default inside the breaking
    window.

---
## 1. Current-state audit

The v1 audit (2026-10-17) is preserved in git history (`git show v1-plan:PLAN.md` is not
tagged — use the commit that merged Phase 9 PR 1, 961e3b5). Durable facts now live in §0 as
decisions. Short version: FastAPI + Postgres/pgvector + SearXNG + trafilatura; tool-calling
pipeline everywhere (`search` deterministic, `research` planner+loop, `agent` autonomous);
model catalog with capabilities + `default_params` passthrough into `init_chat_model`;
API keys + refresh tokens; hybrid retrieval with flashrank; SSE protocol v2 (changelog
through v2.3). Remaining debt relevant to v2: chunk store is global (fixed by Phase 9
PR 6), SearXNG provider lacks per-call `categories` (PR 11), no capability probing for
local models (PR 9), no per-user preferences (PR 10), no jobs layer (PR 13).

---

## 2. Phases

> Convention: **one phase-task = one branch = one PR**. Every step ships its default at
> "current behavior" except where a **deliberate default flip** is noted. Each merged PR
> bumps the minor version (decision 16).

### 2.1 Phase 9 reconciliation (v1 → v2)

| v1 PR | Disposition in v2 |
|---|---|
| ~~PR 1 `wu/hygiene~~ | ✅ shipped (sanity filter on, dedup+fusion, max_urls) |
| PR 2 `wu/cache` | **stays, next up** — v1 scope: search-result TTL cache, Postgres page cache, embed-skip, rule-based `query_variations`; v2 adds: rate limiter onto ICache (behavior unchanged), capability-probe cache consumer (PR 9); LLM answers never cached (decision 17) |
| PR 3 `wu/citations` | stays (after Phase 10; feeds the Note report assembler's citation validator) |
| PR 4 `wu/rewrite` | stays (stands alone; `rewrite: "standalone"` default flip) |
| PR 5 `wu/research-loop` | **absorbed by Phase 11** (reflect–search + brief compression live inside the rebuild presets) |
| PR 6 `wu/recency-scope` | **stays, prerequisite for PR 15** and news source types — v1 scope: dates in prompts and chunk metadata, `recency_bias`, `chunk.conversation_id` + scoping + HNSW + eviction; v2: `scope` ships `global` and **stays global** (decision 21 — no flip; uploads are conversation-scoped unconditionally in PR 15) |
| PR 7 `wu/hybrid` | stays (unscheduled, after rebuild) |
| PR 8 `wu/modes` | **un-deferred by the webapp; absorbed by Phase 11** (presets + protocol v3) |

Recommended order: **PR 2 → PR 6 → Phase 10 (PRs 9–16; uploads/vision intentionally
last per client decision) → Phase 11 (PRs 17–21) → PR 3 / PR 4 / PR 7**. The webapp can
integrate against the current v2.3 protocol immediately (`search` + `research` exist
today) and adopt v3 when Phase 11 lands.

**Cutting vs merging (round 6).** Every branch may be cut from `main` at any time —
`main` already carries all shipped work (961e3b5, no pending branches), so cutting is
unblocked and can happen in parallel. The order above governs **merging**, which must
respect hard dependencies: PR 10 ← PR 9; PR 13 ← PR 12; PR 14 ← PR 12 (offers cache via
PR 2); PR 15 ← PR 6; PR 16 ← PR 9 + PR 15; PR 9 and PR 12 consume `ICache` from PR 2
(soft — land PR 2 first). Phase 11 is sequential (17 → 18 → 19/20 → 21). Area-disjoint
first wave for parallel work: **PR 2, PR 6, PR 11**. Caveat: PR 9–11 and PR 17–21 touch
the same ask-router / SSE-doc files — rebase before merging.

### 2.2 Phase 10 — Webapp client track (PRs 9–16)

- [x] **PR 9 `client/capabilities`** — thinking/effort end-to-end (doubt 2).
  - Version probe per provider: Ollama `/api/show` + `/api/ps` capabilities
    (`completion`, `tools`, `vision`, `thinking`) plus the `think`-levels 400-fallback;
    z.ai effort-level probe (one synthetic call per level, decision 20); cached via
    `ICache` (TTL, e.g. 5 min). OpenRouter model metadata; OpenAI/deepseek catalog seeds
    (no paid-API probing). Probe precedence: **admin override > probe > seed**; probes
    are per catalog entry — a probe result never changes what other providers advertise.
    Admin can override entries.
  - Catalog: `capabilities` extended with `thinking`/`effort`/`vision`; new
    `effort_levels: []` field; `GET /models` exposes both (ModelEntryRead already returns
    capabilities).
  - `ReasoningAdapter` registry per `provider_type`: ollama → `think: true|false` or
    `think: "low"|"medium"|"high"|"max"` (levels first; on HTTP 400 fallback to bool and
    drop `effort_levels` from the cached entry); openai → `reasoning_effort`;
    openrouter → `reasoning: {enabled, effort}`; zai → `thinking: {type: enabled|disabled}`
    + `reasoning_effort: low|medium|high|max` (native mapping, decision 20; GLM-5.3+
    cannot disable thinking → `422 model_lacks_thinking`); deepseek → no runtime
    toggle, **two catalog entries** (chat / reasoner), capabilities empty; unknown → ignore
    + `warnings` entry (degrade, never crash).
  - Request fields `thinking`/`effort` on all ask routes; 422 gating (`model_lacks_thinking`,
    `model_lacks_effort`, `unsupported_effort_level`); factory merge
    `default_params < adapter(request)`.
  - SSE v2.4 (additive): `thinking` event (reasoning-trace passthrough; UI may hide it).
  - Version bump: **0.15.0** (the reset from `1.0.0rc1` happens in this PR's branch, decision 16).
- [x] **PR 10 `client/profile`** — profile preferences (UX round 3).
  - `user_preferences` table (user_id PK, default_model_id, default_mode, thinking,
    effort, source_types, updated_at). `GET/PUT /profile/preferences`;
    `GET /profile/preferences?model_id=X` returns `{stored, resolved, warnings}` —
    resolved = stored values clamped against X's capabilities + X's defaults (this is the
    "model switch resets to defaults" contract, decision 12).
  - Ask routes resolve: request > profile > model default; explicit request fields still
    validated (422 on impossible combos).
- [x] **PR 11 `client/sources`** — source types (doubt 3).
  - `ISearchProvider.search(query, engines, categories)`; SearXNG category map
    (web→general, news→news, academic→science, social→engine list); provider config
    declares `supported_source_types` (DB + type registry, SearchProviderConfig precedent).
  - `GET /search/sources` → types of active/selected provider; ask body gains
    `source_types: []` (validated, unknown → 422; empty = provider default).
  - Personas: `allowed_source_types` intersection left as backlog (v1: provider defaults).
- [x] **PR 12 `client/queue`** — provider-aware resource gates + job queue (round 4:
      cache × inbox × queue; consumes PR 2's ICache).
  - `ResourceGate`: per-resource async semaphore keyed `model:<provider_id>`,
    `search:<provider_id>`, `ggdeals`, …; per-provider `max_concurrency` in provider
    config (Ollama ships 1 — GPU-bound, serial; cloud providers ship ≥ 2 — parallel).
    Foreground requests pass gates too, with bounded wait via `BUSY_POLICY:
    wait|background|reject` (`background` auto-converts a busy foreground ask into a
    queued job → Inbox; `reject` → 503 `provider_busy`).
  - `JobQueue` over the gates: FIFO with per-user fairness (round-robin + per-user running
    cap, default 2) so multiple users never starve each other; external 429s retried with
    backoff honoring `Retry-After`; queue state persisted in the `job` table.
  - Backends: in-process asyncio (default, single-node) or Redis (`REDIS_URL`, optional —
    multi-node, crash-resilient), same lazy pattern as ICache (decision 19).
- [x] **PR 13 `client/inbox`** — jobs layer + Inbox UX (doubt 5; builds on PR 12).
  - `job` table (id, user_id, kind, status queued|running|succeeded|failed|cancelled,
    queue_position, conversation_id, request payload ref, result_summary, error, seen,
    timestamps); `background: true` on research-grade ask routes → `202 {job_id}`;
    `GET /jobs[?seen=]`, `POST /jobs/{id}/seen`, `POST /jobs/{id}/cancel`; per-user SSE
    stream with `job_update` events carrying `status` + `queue_position` (SSE v2.4);
    startup sweep `running → failed`.
  - Inbox view = queue view: pending list (positions), running (progress), done/failed
    (notification, seen flag).
- [x] **PR 14 `client/ggdeals`** — optional, feature-flagged game-offers module (doubt 8;
      official API confirmed 2026-10-28).
  - `src/modules/ggdeals/`: HTTP client (API key in settings, provider-credential
    precedent) + offers TTL cache via ICache; conditional tool attach (`pre_score`
    pattern) behind `GGDEALS_ENABLED` (default off).
  - Tool `game_offers(game_name)` → structured JSON (title, current best offer,
    historical low, stores); ask response carries the structured payload → SSE `offers`
    event → the client renders the banner — never inside answer text. Calls pass the
    `ggdeals` gate (PR 12); exact endpoint shapes, quota and ToS verified at
    implementation.
- [x] **PR 15 `client/uploads`** — text files (doubt 4; client decision 2026-10-28: files
      ship last; depends on PR 6).
  - `POST /conversations/{id}/attachments` (multipart): txt/md/csv (stdlib), pdf (`pypdf`),
    docx (`python-docx`), html (trafilatura). Limits: 10 MB/file, 5 files/request,
    ~50k chars/file extraction cap (configurable). Parsed text → conversation-scoped chunks
    (PR 6 scoping) + `attachment` row.
  - `search_files` tool for the agent (conversation-scoped retrieval); ask body gains
    `attachment_ids`.
  - Scope note (decision 21): **no `scope` default flip here** — upload chunks are
    conversation-scoped unconditionally, web chunks stay global (`scope=conversation`
    stays opt-in); a default flip is deferred to the protocol-v3 window if ever needed.
- [ ] **PR 16 `client/attachments-vision`** — images (round 3 addition; depends on PR 9,
      PR 15).
  - Image allowlist png/jpeg/webp/gif, ≤ 10 MB each, ≤ 4 per request; stored as blobs
    (refs in messages, TTL cleanup job).
  - Ask-time gate: image attachment + model without `vision` → **422 model_lacks_vision**
    (never silent-drop); iron rule ⇒ capabilities ⊇ `{tools, vision}` required.
  - Synthesis prompt builder assembles image content blocks (LiteLLM image format) next to
    the retrieved context; images never enter the search engine path.

---

### 2.3 Phase 11 — Core rebuild: one agentic core, modes = presets (PRs 17–21)

Preset table over the shared loop (toolbox + conversation history + degrade path):

| Preset | Planner | Note | Reflect | Scout | Budget | Role | History |
|---|---|---|---|---|---|---|---|
| `search` | – | – | – | – | low (1–3 iterations) | all | yes |
| `research` | yes | yes | – | – | medium | elevated | yes |
| `deep_research` | yes | yes | yes (`revise_plan`) | yes | high | elevated | yes |
| `persona` | by depth | by depth | by depth | by depth | `persona.default_parameters` (clamped) | elevated | per-persona |

- [ ] **PR 17 `rebuild/core`** — agentic search behind `AGENTIC_SEARCH` (default **off**;
      deliberate flip after eval).
  - Evolved `agent_logic` as the single loop; conversation history moves into the core;
    `POST /ask/search` runs the loop when the flag is on, the deterministic pipeline
    otherwise; `degraded` SSE event on fallback (decision 8).
  - **Eval gate for the flip:** golden set of 30–50 queries; citation fidelity ≥
    deterministic baseline; recall/MRR within −5% relative; p50 latency ≤ 1.5×; token cost
    measured (no gate yet). Flip after 2 consecutive green runs.
- [ ] **PR 18 `rebuild/note`** — `research_note` table + `note_write` (always returns the
      TOC: `{idx, task, one_line_summary, sources_count}`) + `SubmitReport` deterministic
      assembly + SSE `note` event; `research` re-based on the core; degrade →
      gathered→generate_report (decision 9).
- [ ] **PR 19 `rebuild/deep`** — `agent` → `deep_research` (alias `/ask/agent` kept
      temporarily, deprecated); scout step (one forced web_search+fetch+get_context before
      the planner); reflect via `revise_plan(add, rationale)` with ceiling
      `RESEARCH_MAX_SUBQUESTIONS` + slack; high preset budget.
- [ ] **PR 20 `rebuild/persona-mode`** — `POST /ask/persona` (required `persona_id`:
      missing → 422, foreign → 404); `depth: quick|research|deep` (default from persona,
      `_bounded()` clamp); persona-scoped conversations; elevated role gate.
- [ ] **PR 21 `rebuild/protocol-v3`** — POST everywhere (research/agent GET → POST);
      `/ask/agent` → `/ask/deep_research`; scopes `{search, research, deep_research,
      persona}`; one body schema for all ask routes:
      `{query, stream, conversation_id, thinking?, effort?, source_types?, background?,
      attachment_ids?, depth?, persona_id?}`; SSE changelog **v3** (breaking); §0 entry
      recorded; Phase 9 PR 5 + PR 8 closed as absorbed.

### 2.4 Phase 12 — Retrieval upgrades (carried from Phase 9, unscheduled)

- [ ] PR 3 `wu/citations` — numbered `[n]` citations + validator + offline fixture eval
      (feeds the Note assembler's Sources section).
- [ ] PR 4 `wu/rewrite` — history-aware standalone rewriting (default flip:
      `"standalone"`).
- [ ] PR 7 `wu/hybrid` — lexical + vector RRF fusion behind `hybrid_alpha` (default 1.0
      until measured).

---

## 3. Open decisions

| # | Decision | Owner | Needed by |
|---|----------|-------|-----------|
| 1–5 | *(closed 2026-10-17 → 2026-10-26: embedding model, Laya packaging, API-key scheme, `research` role rename, refresh-token design)* | — | closed |
| 6 | ~~Reasoning/effort field naming~~ **Decided 2026-10-28:** `thinking` + `effort`; `reasoning_effort` only as OpenAI wire param inside the adapter | daniel | closed |
| 7 | ~~Profile preferences on model switch~~ **Decided 2026-10-28:** reset thinking/effort to model defaults; request > profile > model default | daniel | closed |
| 8 | ~~Image gating~~ **Decided 2026-10-28:** 422 `model_lacks_vision` at ask time; capabilities ⊇ `{tools, vision}` | daniel | closed |
| 9 | ~~Persona gating~~ **Decided 2026-10-28:** whole persona mode elevated (depth is a budget choice) | daniel | closed |
| 10 | ~~Version baseline~~ **Decided 2026-10-28:** reset to 0.15.0 in PR 9; minor per merged branch; 1.0.0 at webapp GA | daniel | closed |
| 11 | ~~z.ai effort levels~~ **Decided 2026-10-28 (round 5):** native mapping + probe-prune — adapter maps `effort` → `reasoning_effort` (`low|medium|high|max`), catalog seeds per GLM generation, live probe prunes aliased/rejected levels (decision 20) | daniel | closed |
| 12 | ~~gg.deals~~ **Decided 2026-10-28:** official Prices API confirmed (API key; current + historical prices); exact endpoint shapes, quota and ToS verified at PR 14 implementation | daniel | closed |
| 13 | ~~`scope` default flip~~ **Decided 2026-10-28 (round 5):** no flip — web chunks stay global, uploads conversation-scoped unconditionally, `scope=conversation` opt-in; revisit at protocol v3 (decision 21) | daniel | closed |

---

## 4. Backlog (unscheduled)

- **Alternate search backends:** Tavily / SerpAPI / Bing providers (also the "API-keyed
  engines" mitigation for search-egress poisoning, Phase 3 closure).
- **API-caller tool for agent modes:** external REST calls with user-provided (encrypted)
  credentials.
- **Soft delete for user data:** hide-but-preserve (conversations, personas, attachments).
- **Per-model settings memory:** remember last thinking/effort per model in the profile
  (v1 ships single active preference set + reset-on-switch).
- **Persona-level source types:** intersect `allowed_source_types` with provider support.
- **Multi-node topology:** Redis-backed ICache + jobs outgrow single-node asyncio.
- **Job auto-resume:** replay interrupted background runs from the last `research_note`.
- **OCR fallback for non-vision models:** out of scope for v1; would belong in a
  pre-scorer-style sidecar.
