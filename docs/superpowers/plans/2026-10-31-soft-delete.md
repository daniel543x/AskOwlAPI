# Soft delete for user data (`wu/soft-delete`) — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (native, inline).

**Goal:** deleting a conversation or persona hides it everywhere immediately (404, indistinguishable from missing) while the row survives for admin-level recovery.

**Architecture:** nullable `deleted_at` on `conversation` + `persona`; every read/list path filters `deleted_at IS NULL`; continuation paths reject deleted rows with 404; delete endpoints set the timestamp instead of removing rows.

**Tech Stack:** SQLModel + Alembic; pytest offline + integration.

**Spec:** approved short design (2026-10-31, conversation): option A — deleted conversations cannot be continued; no undo in v1.

## Global Constraints

- Undeleted rows behave byte-identically (all filters are additive `deleted_at IS NULL`).
- No route changes anywhere (same paths, same status codes).
- Persona deletion stays non-destructive to conversations: the SET NULL FK keeps working; soft-deleted persona rows remain, so `conversation.persona_id` survives (赡 the deleted persona is itself hidden).
- Branch stacked on `wu/model-settings-memory`; PR → that chain; version `0.<PR>.0`.

## Review Focus

- `GET /history`, `GET /personas`, and detail reads of a deleted row are all 404/absent — six read surfaces, one owner each. → Task 2 tests 1–2.
- Continuing a deleted conversation via any ask route is a 404 before any turn is saved. → Task 2 test 3.
- Deleting a persona keeps its conversations listed in history (they are the user's, not the persona's). → Task 2 test 4.
- A background job whose turn targets a deleted conversation fails cleanly with a sanitized error (existing job error path). → Task 2 test 5.
- Attachment resolution for a deleted conversation rejects too. → Task 2 test 3 (same seam).

---

### Task 1: Columns and delete methods

**Files:**
- Modify: `src/modules/history/model.py` (`Conversation.deleted_at`), `src/modules/personas/model.py` (`Persona.deleted_at`) + one Alembic revision (down_revision `'c2f9e4b8d1a6'`)
- Modify: `src/modules/history/repository.py` (`delete_conversation` sets `deleted_at` and commits, no message/row removals; `get_conversation_for_user` returns None for deleted; `list_conversations` filter)
- Modify: `src/modules/personas/repository.py` (`delete` sets `deleted_at`; `list_for_user`, `get_for_user` filter)
- Test: `tests/modules/test_soft_delete.py` (integration)

- [ ] **Step 1: Failing tests** — delete conversation → row present with deleted_at set, absent from list/detail; delete persona → same; persona delete keeps conversation rows with persona_id intact.
- [ ] **Step 2: RED** (`pytest -m integration`).
- [ ] **Step 3: Implement** columns, methods, filters.
- [ ] **Step 4:** integration + offline suites + migration tests green; commit `Soft-delete conversations and personas with hidden rows`.

### Task 2: Continue/references reject deleted rows

**Files:**
- Modify: `src/core/ask/services/search.py` `_resolve_conversation` (unknown OR deleted → KeyError → 404 path as today) — import nothing new; a deleted row fails the continuation at the same seam attachments resolve at
- Modify: `src/modules/personas/router.py` already relies on `get_for_user` (filtered upstream) — no change expected; assert via test
- Test: `tests/modules/test_soft_delete.py` additions

Tests:
1. `test_deleted_conversation_rejects_continuation` — delete → `POST /ask/search` with conversation_id → 404, nothing saved.
2. `test_deleted_persona_rejects_ask` — delete persona → `/ask/persona` → 404.
3. `test_deleted_persona_conversations_stay_in_history` — history still lists them.
4. `test_background_runner_fails_cleanly` — enqueue job with runner resolving a deleted conversation → job failed with sanitized error (reuse the rerun harness fakes).

- [ ] Steps: RED → implement (expect the persona/persona-router case to already pass — pin it) → suites → commit `Reject deleted rows on continuation paths`.

### Task 3: Bookkeeping

- Tick the PLAN.md backlog line; full suites; PR + version `0.<PR>.0`.

- [ ] **Step 1:** verification + commit + PR.
