# Webapp API batch (persona API surface + jobs bug) — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the accepted asks from the webapp's API notes: persona `description`/`tags`, `persona_id` on history reads + filter, persona `model` with always-wins resolution and the `model_confirm` consent protocol, and the background-job failure fixes (sanitized `Job.error`, re-run duplicate-key investigation).

**Architecture:** One branch, six tasks. Tasks 1–3 add persona fields (SQLModel + Alembic revisions, write-time validation). Task 4 slots a persona tier into chat-model resolution for `POST /ask/persona` and implements the SSE consent round-trip (`model_confirm` event, `allow_model_fallback` body flag) without touching the shared `get_llm_client` chain. Task 5 hardens the job queue's failure surface and adds a re-run regression test. Task 6 updates docs and bookkeeping.

**Tech Stack:** FastAPI + SQLModel/SQLAlchemy async, Alembic, LangChain `BaseChatModel`, pytest (offline unit + live-Postgres `integration` harness).

**Spec:** `ref/API-NOTES.md` (items 1, 2, 4, 5, 9; items 3/6/7/8 are parked or already done). Protocol decisions confirmed in `~/projects/AskOwl/flow.md` ("Backend confirmation — API reply" section) — that file is outside the repo; its confirmed `model_confirm` payload and `allow_model_fallback` placement are binding for Task 4.

## Global Constraints

- English only; `logging` never `print`; never log API keys or decrypted secrets.
- No `HTTPException` raised after SSE streaming started; stream errors are `error` events.
- Tests must pass offline: `uv run pytest`; DB-dependent tests are `pytest.mark.integration` (live test DB only, never the app DB).
- Lint: `uv run ruff check src tests`.
- Every LLM interaction goes through native tool calling; any persona model must carry the `tools` capability.
- Start from a fresh branch off latest `main` (e.g. `wu/persona-api-batch`) **after** `wu/persona-agent-bugs` lands, including its uncommitted `src/tools/db.py` + FK-test changes.
- Schema changes (3 new `persona` columns) must be noted in the PR description.
- The basedpyright errors in `src/core/ask/router.py` / `src/core/research/router.py` / `src/modules/jobs/*` are pre-existing; do not fix or add to them.

## Review Focus

- Old client (never sends `allow_model_fallback`) hits a persona whose model became unusable: turn must end immediately — SSE `model_confirm` then close, no hang, no silent model swap, no conversation/turn persisted. → Task 4 tests 1–2.
- `PATCH /personas/{id}` with an explicit `null` must clear a nullable field (current `exclude_none` makes clearing impossible); omitting the field must not touch it. → Task 1 test 3.
- Over-limit values are rejected, not silently truncated: 9 tags, 51-char tag, 301-char description → 422. → Task 1 test 2.
- `GET /history?persona_id=` must never leak another user's conversations. → Task 2 test 3.
- A failed job's `error` must never carry raw SQL/driver text; full detail stays in server logs. → Task 5 test 1.
- Re-running a background search whose chunks/turn already exist must succeed (this is the reported UniqueViolationError scenario; root cause unconfirmed — the test is the reproduction vehicle). → Task 5 test 3.

---

### Task 1: Persona `description` + `tags`

**Files:**
- Modify: `src/modules/personas/model.py`
- Modify: `src/modules/personas/repository.py:34-45` (`update`)
- Modify: `src/modules/personas/router.py` (no change expected — `Persona(**data.model_dump())` picks new fields up)
- Create: `migrations/versions/d4f8b2e6a9c1_persona_description_tags.py`
- Test: `tests/modules/test_personas_router.py`

**Interfaces:**
- Produces: `Persona.description: str | None = None` (≤300 chars), `Persona.tags: list[str] = []` (max 8 items, each ≤50 chars after `strip()`, empty strings dropped, order preserved); same names/types on `PersonaCreate`, `PersonaUpdate`, `PersonaRead`. Caps are 422 rejections via pydantic, never silent truncation. `PersonaRepository.update` switches to `model_dump(exclude_unset=True)` (drops `exclude_none`) so explicit `null` clears `description`; `tags` clears with `[]`.

- [ ] **Step 1: Write the failing tests** (append to `tests/modules/test_personas_router.py`, reuse `auth_client`/`make_user`/`PERSONA_PAYLOAD`)

1. `test_persona_description_and_tags_roundtrip` — create with `description="Short blurb."`, `tags=["rust", "systems"]`; assert they come back on create, list, get; patch `{"description": "New.", "tags": ["a"]}` → both update.
2. `test_persona_field_caps_reject` — 9 tags → 422; one 51-char tag → 422; 301-char description → 422. Each rejection leaves the persona unchanged (get still returns the original).
3. `test_persona_patch_null_clears_description` — create with description set; patch `{"description": None}` → 200, `description is None`; patch body without `description` → value unchanged.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/modules/test_personas_router.py -q -m integration -k "description or caps or null"`
Expected: FAIL (fields absent → pydantic ignores them; cap test sees no 422).

- [ ] **Step 3: Implement**

- `Persona`: `description: str | None = Field(default=None, max_length=300)`; `tags: list[str] = Field(default_factory=list, sa_column=Column(JSON))`. `PersonaCreate`/`PersonaUpdate`/`PersonaRead` mirror them (`PersonaUpdate` fields optional). Enforce tag rules with a `field_validator("tags")`: strip items, drop empties, then reject >8 items or any item >50 chars with a `ValueError` (FastAPI renders it as 422).
- `PersonaRepository.update`: `data = changes.model_dump(exclude_unset=True)`.
- Migration `d4f8b2e6a9c1` (down_revision `'c3d84b06e9a2'`, follow the header shape of `b8d0e4a6c9f3_persona_scoped_conversations.py`): upgrade adds `persona.description` as `sa.Column('description', sa.String(length=300), nullable=True)` and `persona.tags` as `sa.Column('tags', sa.JSON(), nullable=False, server_default=sa.text("'[]'::json"))`; downgrade drops both. The migration test `tests/modules/test_migrations.py::test_model_metadata_covers_migrated_tables` must stay green.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/modules/test_personas_router.py tests/modules/test_migrations.py -q -m integration`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/modules/personas tests/modules/test_personas_router.py migrations/versions/d4f8b2e6a9c1_persona_description_tags.py
git commit -m "Add persona description and tags with capped schemas"
```

### Task 2: `persona_id` on history reads + `GET /history?persona_id=` filter

**Files:**
- Modify: `src/modules/history/model.py:57-65` (`ConversationRead`)
- Modify: `src/modules/history/repository.py:38-45` (`list_conversations`)
- Modify: `src/modules/history/router.py:23-53`
- Create: `tests/modules/test_history_router.py`

**Interfaces:**
- Produces: `ConversationRead.persona_id: UUID | None` (inherited by `ConversationWithMessages`); `HistoryRepository.list_conversations(self, user_id: UUID, persona_id: UUID | None = None) -> list[Conversation]` — when `persona_id` is given, an equality filter on `Conversation.persona_id` (NULL rows excluded); route query param `persona_id: UUID | None = Query(default=None)`. No `kind`/`mode` field anywhere (per API-NOTES #4).

- [ ] **Step 1: Write the failing tests** (new `tests/modules/test_history_router.py`, `pytestmark = pytest.mark.integration`, harness copied from `auth_client` in `tests/modules/test_personas_router.py`)

1. `test_conversation_read_exposes_persona_id` — insert two `Conversation` rows directly via `db_session` (one with `persona_id`, one without); `GET /history` returns both with correct `persona_id`; `GET /history/{id}` for each returns matching `persona_id`.
2. `test_history_filters_by_persona_id` — `GET /history?persona_id={id}` returns only the persona-scoped row.
3. `test_history_filter_is_user_scoped` — a second user's persona id (with one of the first user's conversations maliciously given that persona_id) returns only rows of the calling user that match; a foreign `persona_id` with no owned rows returns `[]`.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/modules/test_history_router.py -q -m integration`
Expected: FAIL (`persona_id` missing from response; unknown query param ignored).

- [ ] **Step 3: Implement**

- Add `persona_id: UUID | None` to `ConversationRead`. In `router.get_conversation`, pass `persona_id=conversation.persona_id` into `ConversationWithMessages` (it is built field-by-field today).
- Extend `list_conversations` with the optional filter; wire the query param in `list_conversations` route.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/modules/test_history_router.py -q -m integration && uv run pytest -q`
Expected: PASS (offline suite untouched).

- [ ] **Step 5: Commit**

```bash
git add src/modules/history tests/modules/test_history_router.py
git commit -m "Expose persona_id on history reads with a persona filter"
```

### Task 3: Persona `model` column + write-time validation

**Files:**
- Modify: `src/modules/personas/model.py`
- Modify: `src/modules/personas/router.py` (`create_persona`, `update_persona`)
- Modify: `src/modules/llm_provider/selection.py` (add `EntryState` + `unusable_reason`)
- Modify: `src/modules/llm_provider/repository.py` (add `entry_state`)
- Create: `migrations/versions/f1c5e9a3d7b2_persona_model.py`
- Test: `tests/modules/llm_provider/test_selection.py` (offline), `tests/modules/test_personas_router.py` (integration)

**Interfaces:**
- Consumes: `has_role_access(user_roles, required_role)` (selection.py:87).
- Produces:
  - `selection.py`: `@dataclass(frozen=True) class EntryState: id: uuid.UUID; kind: str; is_active: bool; required_role: str; capabilities: list[str]` and `def unusable_reason(state: EntryState | None, user_roles: list[str]) -> str | None` → `None` when usable (kind `"chat"`, active, `"tools"` in capabilities, role accessible), else exactly one of `"missing"` (absent **or** wrong kind), `"inactive"`, `"lacks_tools"`, `"no_role"`.
  - `ModelRepository.entry_state(self, model_id: uuid.UUID) -> EntryState | None` (None when the id is not in the catalog).
  - `Persona.model: uuid.UUID | None` — plain UUID column, **no FK** (a catalog deletion must leave the persona intact and surface as `"missing"` at ask time). Same name on `PersonaCreate`/`PersonaUpdate`/`PersonaRead`.
  - Write path: invalid `model` → HTTP 422 with `detail={"code": "persona_model_invalid", "reason": <token>}`.

- [ ] **Step 1: Write the failing tests**

Offline (`tests/modules/llm_provider/test_selection.py`): `test_unusable_reason_tokens` — table-driven over `EntryState` variants: `None`→`"missing"`, wrong kind→`"missing"`, inactive→`"inactive"`, no `"tools"`→`"lacks_tools"`, role-inaccessible→`"no_role"`, fully valid→`None` (all four checks green).

Integration (`tests/modules/test_personas_router.py`; seed catalog rows following the seeding pattern in `tests/modules/llm_provider/test_repository.py` — a `ModelProvider` plus `ModelCatalogEntry` rows: one active chat-kind entry **with** `"tools"`, one tool-less, one inactive, one non-chat kind):

1. `test_persona_model_rejects_invalid` — create with each bad id → 422, `detail["code"] == "persona_model_invalid"` and the matching `reason`; unknown UUID → reason `"missing"`.
2. `test_persona_model_roundtrip` — create with the valid entry's id → read returns it; patch `{"model": None}` clears it.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/modules/llm_provider/test_selection.py -q && uv run pytest tests/modules/test_personas_router.py -q -m integration -k model`
Expected: FAIL (`EntryState`/`unusable_reason` undefined; router accepts any UUID).

- [ ] **Step 3: Implement**

- `unusable_reason` as pure checks in order missing → inactive → lacks_tools → no_role (order pins the token when several fail).
- `entry_state` reads the entry (+ provider if needed) and maps to `EntryState`.
- Migration `f1c5e9a3d7b2` (down_revision `'d4f8b2e6a9c1'`, same header shape): upgrade adds `persona.model` as `sa.Column('model', sa.Uuid(), nullable=True)`; downgrade drops it. The migration test `tests/modules/test_migrations.py::test_model_metadata_covers_migrated_tables` must stay green.
- Router: inject the llm_provider `ModelRepository` (`Depends(get_repository)` from `llm_provider.repository`); in `create_persona` and `update_persona`, when the incoming `model` is not None, compute `unusable_reason(await repo.entry_state(model_id), [r.name for r in user.roles])` and raise the 422 on a token. Validation runs against the **owner's current roles**.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/modules/llm_provider tests/modules/test_personas_router.py tests/modules/test_migrations.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/modules/personas src/modules/llm_provider tests/modules/llm_provider/test_selection.py tests/modules/test_personas_router.py migrations/versions/f1c5e9a3d7b2_persona_model.py
git commit -m "Add validated persona model field for per-persona chat models"
```

### Task 4: Ask-time persona model resolution + `model_confirm` consent

**Files:**
- Modify: `src/core/ask/services/contracts.py` (`AskBody`)
- Modify: `src/modules/llm_provider/dependency.py` (extract `turn_resources_for_resolved`)
- Create: `src/core/research/services/persona_model.py`
- Modify: `src/core/research/router.py` (`ask_persona`)
- Test: `tests/core_research/test_persona_model_resolution.py` (new, offline), `tests/modules/test_personas_router.py` (integration)

**Interfaces:**
- Consumes: `unusable_reason`/`EntryState` (Task 3), `build_chat_model` (dependency.py:68), `ModelRepository.resolve_model` (repository.py:50), `model_gate`/`search_gate` (jobs/gates.py), `_sse_frame` (core/ask/services/search.py:242).
- Produces:
  - `AskBody.allow_model_fallback: bool = False` (inherited by `PersonaBody`; sits next to `depth`/`persona_id` per the confirmed protocol).
  - `dependency.py`: `def turn_resources_for_resolved(resolved: ResolvedModel, searching: ISearchProvider) -> TurnResources` — the model gate (`resolved.provider_id`, `resolved.max_concurrency`) + search gate + `model_capabilities`, exactly the body of `get_turn_resources` lines 185-196.
  - `persona_model.py`:
    - `@dataclass(frozen=True) class ModelConfirm: persona_model: str; reason: str; fallback_model: str | None` with `def payload(self) -> dict[str, str | None]`.
    - `async def resolve_persona_model(persona, allow_model_fallback: bool, *, fallback_model: BaseChatModel, fallback_turn: TurnResources, repo: ModelRepository, factory: ChatModelFactory, user: User, profile: UserPreferences | None, thinking: bool | None, effort: str | None, searching: ISearchProvider) -> tuple[BaseChatModel, TurnResources] | ModelConfirm`
    - Decision table (the heart of this task):
      - `persona.model is None` → `(fallback_model, fallback_turn)` (today's behavior).
      - reason := `unusable_reason(await repo.entry_state(persona.model), roles)`; if reason is not None and not `allow_model_fallback` → `ModelConfirm(persona_model=str(persona.model), reason=reason, fallback_model=<resolved>)` where `<resolved>` = `str(r.entry_id)` from `repo.resolve_model(kind="chat", user_id=user.id, user_roles=roles, requested_model_id=None, profile_model_id=profile.default_model_id if profile else None)`, or `None` when the builtin tier has no `entry_id`.
      - reason is not None but consent given → `(fallback_model, fallback_turn)` (the standard chain the route dependency already built).
      - reason is None → rebuild: `model = await build_chat_model(repo, factory, user, request_model_id=persona.model, thinking=thinking, effort=effort, profile=profile)` (request tier ⇒ persona wins, tools capability re-checked, reasoning validated) and `turn = turn_resources_for_resolved(await repo.resolve_model(... requested_model_id=persona.model ...), searching)`. The double `resolve_model` call is accepted: it keeps `build_chat_model` untouched.
  - `ask_persona` flow order (binding): load persona (404) → **persona-model block** → depth resolution → `ensure_source_types` → attachments → `ensure_vision(turn.model_capabilities, ...)` (so vision is judged against the persona model) → dispatch. When `resolve_persona_model` returns a `ModelConfirm`: `stream=True` (default) → `StreamingResponse` yielding exactly one `_sse_frame(1, "model_confirm", confirm.payload())` then closing, nothing persisted; `stream=False` → `HTTPException(409, detail={"code": "persona_model_unusable", **confirm.payload()})`; `background=True` → same 409 before enqueueing (consent can never be asked mid-job).

- [ ] **Step 1: Write the failing tests**

Offline (`tests/core_research/test_persona_model_resolution.py`; fakes: a stub repo with `entry_state`/`resolve_model`, a stub factory returning a sentinel client, `SimpleNamespace` persona/user/profile, reuse `TurnResources` from `jobs.gates`):

1. `test_persona_model_none_keeps_fallback` — persona without model → returns the injected fallback client/turn unchanged.
2. `test_unusable_model_without_consent_yields_confirm` — entry_state inactive → `ModelConfirm` with `reason="inactive"`, exact payload keys `persona_model`/`reason`/`fallback_model`.
3. `test_unusable_model_with_consent_falls_back` — same state, `allow_model_fallback=True` → injected fallback client.
4. `test_usable_model_rebuilds_client_and_turn` — valid chat entry → returned client comes from the factory with the persona entry's id, turn keys carry the persona provider gate id.

Integration (`tests/modules/test_personas_router.py`; the `auth_client` harness already overrides `get_llm_client` with `ScriptedToolModel` — the confirm path fires before that model is used, and the consent path runs it):

5. `test_persona_ask_model_confirm_stream` — persona with `model=<random UUID>`, `POST /ask/persona` streaming → first SSE event is `model_confirm` with `reason="missing"`, `fallback_model` present or null, no `done`/`message` events after, and no turn saved (the conversation has no messages).
6. `test_persona_ask_model_confirm_json_409` — same body with `stream=false` → 409, `code == "persona_model_unusable"`.
7. `test_persona_ask_background_model_confirm_409` — `depth="research"`, `background=true` → 409, nothing enqueued (`GET /jobs` empty).
8. `test_persona_ask_allow_model_fallback_runs` — same missing model, `allow_model_fallback=true`, depth `quick` → normal run: `message` + `done` events arrive.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/core_research/test_persona_model_resolution.py -q && uv run pytest tests/modules/test_personas_router.py -q -m integration -k confirm`
Expected: FAIL (module undefined; unknown body flag; route runs the turn instead of confirming).

- [ ] **Step 3: Implement**

Per the Interfaces block: extract `turn_resources_for_resolved` (have `get_turn_resources` delegate to it — one-line change, no behavior shift), add `AskBody.allow_model_fallback`, write `persona_model.py`, and wire the block into `ask_persona` at the pinned position. The confirm frame must be the stream's only frame; do not create a conversation or acquire gates in that path.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/core_research tests/modules/test_personas_router.py -q` then `uv run pytest -q && uv run ruff check src tests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/ask/services/contracts.py src/modules/llm_provider/dependency.py src/core/research tests/core_research/test_persona_model_resolution.py tests/modules/test_personas_router.py
git commit -m "Resolve persona chat model first with model_confirm consent flow"
```

### Task 5: Jobs — sanitized `Job.error`, tolerant submit persist, re-run regression test

**Files:**
- Modify: `src/modules/jobs/queue.py:86-95` (`submit`), `:184-190` (`_run` failure branch)
- Modify: `tests/modules/jobs/test_queue.py`
- Create: `tests/modules/jobs/test_rerun_regression.py`

**Interfaces:**
- Produces: failure path sets `job.error = f"Job failed with {type(exc).__name__}."` (full traceback stays server-side via the existing `logger.exception`); `submit` tolerates `_persist` failures — log a warning, keep the job in the in-memory queue, still return the row (fixes the observed "500 but job enqueued anyway"; the row backfills on the next transition's `_persist`).

- [ ] **Step 1: Write the failing tests**

In `tests/modules/jobs/test_queue.py` (offline harness already there — no `session_factory`):

1. `test_failed_job_error_is_sanitized` — runner raises `RuntimeError('duplicate key value violates unique constraint "chunk_content_hash_key"')` → job fails with `error == "Job failed with RuntimeError."`; assert `"duplicate"` not in `job.error`.
2. `test_submit_survives_persist_failure` — `session_factory` that raises on use → `submit` returns the queued job instead of raising, worker still runs it to `succeeded`.

New `tests/modules/jobs/test_rerun_regression.py` (`pytest.mark.integration`; the reproduction vehicle for API-NOTES #9): enqueue through a real `JobQueue(session_factory=...)` a runner that calls `search_json_adapter` twice with the **same query** against the live test DB (fakes from `tests/core_ask/fakes.py`: `FakeSearchProvider`, `make_scraper`, `FakeRanker`, `embedder=None`), asserting both jobs reach `succeeded` and the second `store_chunks` call inserts 0 rows.

3. `test_background_search_rerun_succeeds`

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/modules/jobs/test_queue.py -q && uv run pytest tests/modules/jobs/test_rerun_regression.py -q -m integration`
Expected: tests 1–2 FAIL (`str(exc)` currently stored; persist failure propagates). Test 3 documents today's behavior: if it FAILS, the UniqueViolationError is reproduced — read the constraint name from the error and fix that write path (all current chunk/page writes already use `ON CONFLICT`, so a failure implicates a path without conflict handling or a transaction-interleaving race; fix at the constraint's table, then this test is the regression pin). If it PASSES, ship it as the guard and note in the PR that the live-DB root cause remains open, diagnosable from `logger.exception` output now that `Job.error` no longer truncates context.

- [ ] **Step 3: Implement**

Queue changes per Interfaces. Keep the retry branch untouched (rate-limit retries store no error).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/modules/jobs -q && uv run pytest -q && uv run ruff check src tests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/modules/jobs/queue.py tests/modules/jobs
git commit -m "Sanitize job errors, tolerate persist failures, pin rerun behavior"
```

### Task 6: Docs, PLAN.md, and API-NOTES bookkeeping

**Files:**
- Modify: `docs/api/ask_research_sse.md`
- Modify: `PLAN.md` (new phase section under `## 2. Phases`; keep `## 3. Open decisions` current)
- Modify: `ref/API-NOTES.md` (local, untracked — statuses only, not committed)

**Interfaces:** none (documentation).

- [ ] **Step 1: Document `model_confirm`** in `docs/api/ask_research_sse.md`: the event payload (`persona_model`, `reason` ∈ `missing`/`inactive`/`lacks_tools`/`no_role`, `fallback_model` advisory — raw catalog entry ids resolvable via `GET /models?kind=chat`), the re-submission contract (`allow_model_fallback: true` in the body), the 409 JSON shape for `stream=false`/`background`, and one line noting `model_confirm` is persona-only. `debug` is already documented (API-NOTES #7 closed).
- [ ] **Step 2: PLAN.md** — add the phase section summarizing this batch with its checklist ticked, and note the schema change (persona: `description`, `tags`, `model`) for the PR description.
- [ ] **Step 3: `ref/API-NOTES.md`** — flip #1, #2, #4, #5 to landed with the date, move #9 to fixed-if-applicable (or keep open with the re-run test reference if the live-DB root cause is still unconfirmed).
- [ ] **Step 4: Full verification**

Run: `uv run pytest -q && uv run pytest -m integration -q && uv run ruff check src tests`
Expected: PASS (integration suite requires the live test DB via `podman-compose up -d db`).

- [ ] **Step 5: Commit**

```bash
git add docs/api/ask_research_sse.md PLAN.md
git commit -m "Document model_confirm protocol and record the webapp API batch"
```
