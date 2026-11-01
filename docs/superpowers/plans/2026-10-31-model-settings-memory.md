# Per-model settings memory (`wu/model-settings-memory`) — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (native, inline).

**Goal:** the profile remembers the last thinking/effort per model and restores it on switch.

**Architecture:** new `usermodelsettings` table (PK user_id + model_catalog_entry_id); `PUT /profile/preferences` upserts memory when thinking/effort are explicit, and on an actual model switch with both omitted restores the remembered (re-clamped) values instead of the model defaults. resolved view, GET route, ask paths, and API shapes unchanged.

**Tech Stack:** SQLModel + Alembic; pytest offline + integration.

**Spec:** approved short design (2026-10-31, conversation); backlog "per-model settings memory", supersedes the reset-on-switch behavior only when memory exists.

## Global Constraints

- Decision 12 stays true: asks stay stateless; the profile row is the single resolved source.
- A user with no memory rows must see byte-identical behavior to today.
- clamped values are stored (memory never holds values the model rejects).
- Branch stacked on `wu/persona-source-types`; PR → that chain; version `0.<PR>.0` (CRUSH.md).

## Review Focus

- Explicit thinking/effort wins AND refreshes memory (clamped). → Task 1 test 3.
- Switch back to a remembered model restores its values, re-clamped to current capabilities. → Task 1 test 2.
- Same-model write with omitted thinking/effort keeps today's default-reset semantics (no surprise memory). → Task 1 test 4.
- Memory for a deleted/inactive model never leaks into views (restore only via the effective-model path, which clears unknown models anyway). → Task 1 test 5.

---

### Task 1: Table, repository, memory flow

**Files:**
- Modify: `src/modules/profile/model.py` (`UserModelSettings`, table) + migration (`down_revision 'a9d3e7c1b5f8'`)
- Modify: `src/modules/profile/repository.py` (`ModelSettingsRepository.get(user_id, model_id)`, `.upsert(user_id, model_id, thinking, effort)`; note the existing PreferencesRepository.set/update pattern)
- Modify: `src/modules/profile/router.py` (`put_preferences`: upsert-when-explicit; restore-on-actual-switch when both omitted — differing from `stored.default_model_id` first, else the stored row)
- Modify: `src/modules/profile/factory.py` (new `get_settings_repository` dependency)
- Test: `tests/modules/profile/test_settings_memory.py` (integration; seed catalog entries like `tests/modules/llm_provider/test_repository.py`)

Tests:
1. `test_no_memory_keeps_default_reset` — PUT model A without thinking → view shows model A defaults (thinking None or model default), no memory row written.
2. `test_switch_back_restores_remembered_and_reclamps` — PUT A thinking=true/effort=low (rows written); PUT B (defaults); PUT back A without fields → thinking/effort restored as saved; if a second entry changes capabilities, clamped accordingly (assert via catalog entries with different capabilities).
3. `test_explicit_wins_and_refreshes` — PUT back A WITH thinking=false → view false, memory updated to false.
4. `test_same_model_write_without_fields_resets` — PUT A again without fields (after memory) → view shows defaults, memory row UNCHANGED.
5. `test_memory_for_cleared_model_ignored` — PUT default_model_id unknown → cleared-warning + defaults; the stale memory rows for other models never leak.

- [ ] Steps: RED (tests reference not-yet-existing repo methods/table) → implement → `uv run pytest tests/modules/profile -q -m integration && uv run pytest -q` → commit `Remember thinking and effort per model in the profile`.

### Task 2: Bookkeeping

- Tick the PLAN.md backlog line → commit → PR (schema note: new `usermodelsettings` table) → version `0.<PR>.0`.

- [ ] **Step 1:** `uv run pytest -q && uv run pytest -m integration -q && uv run ruff check src tests` → commit → PR + version.
