# Persona agent bug fixes — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> (native, inline). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a persona agent run survives free-form model output, a `status`
heartbeat streams during long LLM calls, and persona deletion no longer 500s.

**Architecture:** three independent fixes in one bug branch: (1) agent-loop
salvage (one nudge round, then stream prose as the final answer), (2) an
asyncio heartbeat wrapper around `bound.ainvoke`, (3) `conversation.persona_id`
FK `ON DELETE SET NULL` (model + idempotent bootstrap ALTER, no Alembic).

**Tech Stack:** FastAPI + SQLModel/SQLAlchemy async, LangChain `BaseChatModel`,
pytest (offline unit + live-Postgres integration harness in `tests/conftest.py`).

**Spec:** `docs/superpowers/specs/2026-10-29-persona-agent-bugs-design.md`

## Global Constraints

- English only; `logging`, never `print`; no `HTTPException` after streaming.
- Async end-to-end: no sync blocking calls inside async functions.
- Tests offline except `pytest.mark.integration` (live test DB only — never the
  app DB; conftest guard enforces `askowl_test`).
- Commands: `uv run pytest`, `uv run ruff check src tests`.
- Schema change must be noted in the PR description (create_all cannot alter
  existing tables; the bootstrap ALTER is the migration).

## Review Focus

- Prose-only response with empty content (e.g. a reasoning-only message that
  produced no text): nudge must not re-fire forever; test pins one error after
  the second bare pass. → Task 1 test 2.
- `response.content` as a list of content blocks (reasoning models): must join
  text blocks, never `str(list)`. → Task 1 test 3.
- A single free-form response again mid-run must not consume the whole budget
  in nudge loops: exactly one nudge per run. → Task 1 test 1 asserts message
  count of 2 modell calls max before salvage.
- Persona with zero conversations still deletes (FK change must not regress
  the clean case). → Task 3 integration test.
- The bootstrap ALTER is idempotent and safe on a fresh DB where the table is
  already created with `SET NULL`. → Task 3 reruns `init_db` twice.

---

### Task 1: Agent-loop free-form salvage (`agent.py`)

**Files:**
- Modify: `src/core/research/services/agent.py:141-176` (loop body)
- Test: `tests/core_research/test_agent_loop.py`
- Reuses: `tests/core_ask/fakes.py` `ScriptedToolModel` (records `self.calls`
  = the message lists sent to the model; scripted `AIMessage`s popped in order)

**Interfaces:**
- Produces: module constant `NUDGE = "You MUST respond now by calling the SubmitOutcome tool."`, and `_text_of(content: Any) -> str` joining `{"type": "text"}` blocks when `content` is a list. Behavior contract unchanged otherwise: events are `("message", {"text": ...})` final answers and `("done", {"status": "completed"})`.

- [ ] **Step 1: Rewrite the failing tests**

  Replace `test_agent_neever_accepts_free_form_responses` with:

  1. `test_agent_salvages_repeated_free_form_as_final_answer` — script
     `[AIMessage("prose one"), AIMessage("prose two")]`, budget
     `Budget(max_iterations=8, time_budget_seconds=60.0, token_budget=100_000)`;
     assert no `error`, `messages == ["prose two"]`, `done == [{"status":
     "completed"}]`, and `len(model.calls) == 2` (nudge appended exactly once;
     `model.calls[1]` contains `NUDGE` as its last message).
  2. `test_agent_bare_second_free_form_still_errors` — script
     `[AIMessage("prose one"), AIMessage(content="")]`; assert an `error` event
     mentioning free-form output and no `done`.

  Task 2 extends test 1's model with a delay; keep tests 1–3 single-file.

- [ ] **Step 2: Run to verify failure**

  Run: `uv run pytest tests/core_research/test_agent_loop.py -x -q`
  Expected: new tests FAIL on the current error-at-first-prose behavior.

- [ ] **Step 3: Implement the salvage branch**

  In the loop body after `outcome is None` (so: no tool calls at all),
  replacing the current hard `error`+`return`:
  - Text = `_text_of(response.content)` (helper joins `"text"` blocks when
    `content` is a list, else `str(content)`).
  - First bare prose **with text**: append `response`, append
    `HumanMessage(NUDGE)`, `budget.spend_iteration()`, `continue`.
  - Bare prose again (or empty text): if text → yield
    `("message", {"text": text})`, `("done", {"status": "completed"})`,
    `return`; else yield the existing free-form `error` event unchanged and
    `return`.

- [ ] **Step 4: Run tests to verify pass + full offline suite**

  Run: `uv run pytest tests/core_research -q && uv run pytest -q`
  Expected: PASS (note the pre-existing 5 basedpyright errors in
  `src/core/research/router.py` are out of this branch's scope).

- [ ] **Step 5: Commit**

```bash
git add src/core/research/services/agent.py tests/core_research/test_agent_loop.py
git commit -m "Salvage persona agent runs from free-form model output"
```

### Task 2: Status heartbeat during the LLM call (`agent.py`)

**Files:**
- Modify: `src/core/research/services/agent.py` (loop's `bound.ainvoke` site)
- Test: `tests/core_research/test_agent_loop.py`

**Interfaces:**
- Produces: module constant `HEARTBEAT_SECONDS = 15.0` and
  `async def _beaten(model: BaseChatModel, messages: list) -> AsyncGenerator[PipelineEvent]`
  yielding `("status", {"step": "Agent thinking… ({n}s)"})` ticks
  (n = elapsed seconds floored) while `model.ainvoke(messages)` pends, then a
  final `("__llm_done__", AIMessage)` sentinel. Call site filters status ticks
  through and captures the response from the sentinel; the Task 1 salvage
  logic then consumes that message unchanged.

- [ ] **Step 1: Write the failing test**

  `test_status_heartbeat_during_long_llm_call` — define `SlowModel` in
  `tests/core_research/test_agent_loop.py` subclassing `BaseChatModel` whose
  `agenerate` awaits `asyncio.sleep(0.05)` then returns
  `tool_call("SubmitOutcome", {"answer": "ok."})`, piped through
  `_convert_to_generation` (build `ChatResult(generations=[ChatGeneration(
  message=response)])`). Monkeypatch `agent.HEARTBEAT_SECONDS` to `0.01` via
  `import ...agent as _a; monkeypatch.setattr(_a, "HEARTBEAT_SECONDS", 0.01)`.
  Assert: ≥1 `status` step contains "Agent thinking", plus the normal
  `message`/`done` events, and no `error`.

- [ ] **Step 2: Run to verify failure**

  Run: `uv run pytest tests/core_research/test_agent_loop.py::test_status_heartbeat_during_long_llm_call -q`
  Expected: FAIL (no heartbeat statuses exist; the spawn helper is undefined).

- [ ] **Step 3: Implement `_beaten`**

```python
async def _beaten(model, messages):
    """`ainvoke` yielding a status heartbeat while the call is pending."""
    started = monotonic()
    task = asyncio.create_task(model.ainvoke(messages))
    while True:
        try:
            response = await asyncio.wait_for(asyncio.shield(task),
                                              HEARTBEAT_SECONDS)
            return response
        except TimeoutError:
            elapsed = floor(monotonic() - started)
            yield "status", {"step": f"Agent thinking… ({elapsed}s)"}
```

  At the call site: `async for`-style is awkward inside the existing loop —
  use `response = None; async for chunk in _beaten(...): yield chunk` pattern
  or restructure as an explicit wrapper per the generator semantics (the
  implementer picks the minimal-looking form; the tests own behavior).

- [ ] **Step 4: Run the loop test file, then the offline suite**

  Run: `uv run pytest tests/core_research -q && uv run pytest -q && uv run ruff check src tests`
  Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/research/services/agent.py tests/core_research/test_agent_loop.py
git commit -m "Emit a status heartbeat while the agent LLM call is pending"
```

### Task 3: Persona delete preserves history (`history/model.py` + bootstrap)

**Files:**
- Modify: `src/modules/history/model.py:27-29` (`Conversation.persona_id` →
  `sa_column=Column(ForeignKey("persona.id", ondelete="SET NULL"),
  nullable=True, index=True)`)
- Modify: `src/tools/db.py` (`init_db` — idempotent constraint swap after
  `create_all`)
- Test: `tests/modules/test_personas_router.py` (integration, marked)

**Interfaces:**
- Consumes: existing `test_personas_router.py` harness (`auth_client`, the
  direct `db_session`). Produces: constraint `conversation_persona_id_fkey`
  with `confdeltype = 'a'` (SET NULL) after `init_db`, and a 204 on
  `DELETE /personas/{id}` for personas owning conversations.

- [ ] **Step 1: Write the failing integration tests**

  In `tests/modules/test_personas_router.py` (pytestmark already
  `integration`):

  1. `test_delete_persona_with_conversations_preserves_history` — create
     persona + one `Conversation(persona_id=persona.id)` via `db_session`;
     `await auth_client.delete(f"/personas/{persona.id}")`; assert 204 and the
     conversation row still exists with `persona_id is None`.
  2. `test_init_db_swap_is_idempotent` — after the conftest fixture has already
     `create_all`ed the schema, inspect
     `pg_constraint.confdeltype` for `conversation_persona_id_fkey` == `'a'`
     (SET NULL), run `init_db(engine=...)` again on the fixture engine, assert
     no exception and `confdeltype` still `'a'`.

- [ ] **Step 2: Run to verify failure**

  Run: `uv run pytest tests/modules/test_personas_router.py -m integration -q`
  Expected: test 1 FAILS with `ForeignKeyViolationError` (reproduced 2026-10-29);
  test 2 FAILS with `confdeltype == 'n'` (NO ACTION).

- [ ] **Step 3: Implement**

  - `model.py`: replace the `Field` with a `sa_column` FK carrying
    `ondelete="SET NULL"` (follow the module's existing `Column(JSON)` style).
  - `db.py` `init_db`, after `run_sync(SQLModel.metadata.create_all)`:

    a single `DO $$` block guarded by
    `SELECT conname FROM pg_constraint WHERE conname = 'conversation_persona_id_fkey' AND confdeltype <> 'a'`:
    `ALTER TABLE conversation DROP CONSTRAINT conversation_persona_id_fkey;
     ALTER TABLE conversation ADD ... FOREIGN KEY (persona_id) REFERENCES persona(id) ON DELETE SET NULL;`
    No-op on fresh DBs (constraint already `'a'`); note in the PR: Schema — no
    Alembic yet, existing DBs migrated in the bootstrap (CRUSH.md convention).

- [ ] **Step 4: Run integration tests, then the whole suite + lint**

  Run: `uv run pytest tests/modules/test_personas_router.py -m integration -q && uv run pytest tests/core_research -q && uv run ruff check src tests`
  Expected: PASS (integration tests skip offline below the repo's PASS bar;
  with Postgres up they run).

- [ ] **Step 5: Commit**

```bash
git add src/modules/history/model.py src/tools/db.py tests/modules/test_personas_router.py
git commit -m "Preserve persona conversations when a persona is deleted"
```

### Task 4: Design doc + final assembly

**Files:**
- Create (already on disk, untracked):
  `docs/superpowers/specs/2026-10-29-persona-agent-bugs-design.md`
- Modify: `.gitignore` (already modified: ignore the `flow.md` symlink)

- [ ] **Step 1: Commit the design doc and gitignore with the branch**

```bash
git add docs/superpowers/specs/2026-10-29-persona-agent-bugs-design.md .gitignore
git commit -m "Add design notes for the persona agent bug phase"
```

- [ ] **Step 2: Whole-suite verify before handoff**

  Run: `uv run pytest -q && uv run ruff check src tests && uv sync --all-extras --check`
  Expected: all PASS.

The user's stashed `PLAN.md` work (git stash: "hybrid PLAN.md WIP") belongs
to `wu/hybrid-v2` and stays stashed; it is not part of this branch.
