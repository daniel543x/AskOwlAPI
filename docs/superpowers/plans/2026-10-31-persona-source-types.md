# Persona-level source types (`wu/persona-source-types`) — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (native, inline). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a persona can pin the web-slice its turns search (`allowed_source_types`), intersected with the request and the provider's declared set.

**Architecture:** one persona column + write-time validation (mirroring persona `model`), and a single ask-time narrowing point in `ask_persona` whose effective list feeds all four dispatch paths — no new toolbox parameter.

**Tech Stack:** FastAPI + SQLModel, Alembic, pytest offline + integration.

**Spec:** the approved short design in conversation (2026-10-31); backlog item "persona-level source types" in `PLAN.md` §4.

## Global Constraints

- Empty persona list = unconstrained; empty effective list = provider defaults; both preserve today's byte-level behavior.
- A typo fails loudly at write (422); provider drift at ask time degrades (filter + warning), never crashes a run.
- Commands: `uv run pytest`, `uv run pytest -m integration`, `uv run ruff check src tests`.
- Branch stacked on `wu/numbered-citations`; PR → `wu/numbered-citations`; version `0.<PR>.0` after `gh pr create` (CRUSH.md).
- basedpyright pre-existing `src/core/**` errors out of scope.

## Review Focus

- A request naming a type the persona forbids → 422 `persona_source_type_mismatch`, run never starts. → Task 2 test 2.
- Persona values no longer declared by the provider are filtered with a warning, not a 422. → Task 2 test 4.
- Request fills the gap: request ∩ persona when both non-empty; persona alone when request empty. → Task 2 test 1.
- All four dispatch paths (quick / SSE research / JSON research / background) receive the SAME effective list. → Task 2 tests 1–3.

---

### Task 1: Persona field, write-time validation, migration

**Files:**
- Modify: `src/modules/personas/model.py` (`Persona`, `PersonaCreate`, `PersonaUpdate`, `PersonaRead` + `allowed_source_types`)
- Modify: `src/modules/personas/router.py` (validate on create/update: declared-set check → 422 `{"code": "persona_source_type_invalid", "types": [...]}`)
- Modify: `src/modules/search_engine/source_types.py` — none; reuse `get_declared_source_types` (search_engine.factory) as a router dependency
- Create: `migrations/versions/<gen>_persona_allowed_source_types.py` (down_revision `'f1c5e9a3d7b2'`, JSON column nullable=False server_default `'[]'`)
- Test: `tests/modules/test_personas_router.py`

- [ ] **Step 1: Failing tests** — `test_persona_source_types_crud` (create with types → read round trip; patch `{"allowed_source_types": []}` clears; patch without the field unchanged) and `test_persona_source_types_reject_undeclared` (create with `"not_a_type"` → 422 code `persona_source_type_invalid` listing it; valid declared types pass; empty list passes). Seed a provider/declared set like Task 3 of the API batch did, or monkeypatch `get_declared_source_types` in the harness.
- [ ] **Step 2: Run RED.** Expected: field ignored, unknown values accepted.
- [ ] **Step 3: Implement** field + router validation using the declared set the route resolves as a dependency.
- [ ] **Step 4:** `uv run pytest tests/modules/test_personas_router.py tests/modules/test_migrations.py -q -m integration` → PASS.
- [ ] **Step 5:** Commit `Add persona allowed_source_types with declared-set validation`.

### Task 2: Ask-time narrowing across all dispatch paths

**Files:**
- Modify: `src/core/research/router.py` (`ask_persona`: compute `effective` after persona load + source-type validation; 422 `{"code": "persona_source_type_mismatch", "types": [...]}`; use `effective` in `_persona_quick_turn`'s toolbox (`source_types=`), and in the four dispatch sites that currently pass `request.source_types`)
- Modify: `src/core/research/router.py` `_persona_quick_turn` signature (+`source_types`, passed to `AskToolbox`)
- Test: `tests/core_research/test_persona_source_types.py` (offline unit: `_effective_source_types(request, persona, declared)` pure function) + `tests/modules/test_personas_router.py` (integration: quick + research SSE carry persona slice)

- [ ] **Step 1: Failing offline tests** for the pure function: request ∩ persona; persona alone; request alone (persona empty); request outside persona → `SourceTypeMismatch` exception carrying the offending types; declared drift filters + marks a warning (capture via caplog).
- [ ] **Step 2: RED.** Expected: function undefined.
- [ ] **Step 3: Implement** `_effective_source_types` + wiring at the four sites; drift case: effective = list ∩ declared with `logger.warning`.
- [ ] **Step 4:** `uv run pytest -q` and `uv run pytest tests/modules/test_personas_router.py -q -m integration` (one integration test: persona with source_types → POST /ask/persona quick → the search provider received only that category) → PASS; ruff.
- [ ] **Step 5:** Commit `Narrow persona asks to the persona's source types`.

### Task 3: Bookkeeping

- Tick PLAN.md backlog entry (`persona-level source types`), PR body notes the schema change (new persona column), version `0.<PR>.0` after `gh pr create`.

- [ ] **Step 1:** PLAN.md tick + full verification (`uv run pytest -q && uv run pytest -m integration -q && uv run ruff check src tests`) → commit `Record the persona source types batch`.
