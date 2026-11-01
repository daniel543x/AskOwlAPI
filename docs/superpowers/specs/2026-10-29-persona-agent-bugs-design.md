# Bug phase design — persona agent abort + persona delete 500 + status heartbeat

Scope: API-NOTES items 6 (a+b) and webapp #8 (`DELETE /personas/{id}` 500). One
branch, one PR, per the one-phase-one-branch convention. New branch off main
(current branch `wu/hybrid-v2` carries separate in-flight work).

## Bug 1 — persona agent loop dies on free-form model output (item 6a)

**Evidence / root cause:** `agent_logic` (`src/core/research/services/agent.py`
158–169) requires every response to contain an executable tool call. When the
model returns prose mid-run (observed ~90 s in, after successful tool rounds),
the run terminates with `error: "The agent responded without a tool call…"` and
all gathered context is lost.

**Fix (treat-free-form-as-final with one nudge):**
1. First prose-only response mid-run: append a short HumanMessage nudge
   ("You MUST respond by calling SubmitOutcome…") and continue the loop,
   consuming one iteration of budget.
2. Second consecutive prose response (or prose when budget is nearly out):
   stream `response.content` as the final answer (`message` event), then
   `done` — the run is salvaged, not lost.
3. Rationale for "answer as-is" over forcing `_stream_forced_outcome`: the
   prose already exists on the current response; a second synthesis call over
   gathered context is slower and can hit the same contract fault. The loop
   prompt already demands citations, so quality is comparable.

## Bug 2 — `DELETE /personas/{id}` 500 (webapp API-NOTES #8)

**Evidence / root cause (reproduced on live test Postgres):** the
`conversation.persona_id` FK (`src/modules/history/model.py:27`) is created
without an `ON DELETE` action → Postgres `NO ACTION`. Deleting a persona that
has persona conversations raises `ForeignKeyViolationError on "persona"`,
surfacing as a 500. (Not a MissingGreenlet lazy-load, as suspected.)

**Fix (root: FK becomes `ON DELETE SET NULL`):**
- Owner data is preserved: persona conversations survive as ordinary history
  (`persona_id` → NULL), matching the item-4 routing rule already agreed.
- Model change: `Conversation.persona_id` FK with
  `ondelete="SET NULL"` for fresh databases.
- Existing databases: idempotent `information_schema`-guarded
  `ALTER TABLE ... DROP CONSTRAINT ... ADD ... ON DELETE SET NULL` in the
  `init_db` bootstrap (no Alembic yet; schema change flagged in PR text).
- Alternatives rejected: 409-until-conversations-deleted (degrades UX, no
  bulk history delete exists), CASCADE-delete conversations + messages
  (destroys user data; soft delete is a backlog item).

## Bug 3 — silent stream during long LLM calls (item 6b, part of item 6)

**Fix:** while `bound.ainvoke` is pending, emit a `status`
(`{"step": "Agent thinking…"}`) roughly every 15 s from an asyncio side task;
cancelled on completion. Products: UI stays alive; unknown events are already
ignored by today's clients. Interval derived from nothing magic — 15 s is well
below proxy read timeouts and the ~90 s stall observed.

## Tests

- Unit (offline, `tests/core_research/test_agent_loop.py` fakes): prose mid-run
  → nudge once → prose again → answer + `done` (no `error`); heartbeat emits
  during a slow scripted model response.
- Integration (live test DB, marked `integration`): persona with a persona
  conversation deletes cleanly; conversation stays with `persona_id IS NULL`.

## Out of scope (later phases)

Schema batch (description/tags/persona model + consent), history filter, SSE
contract doc, generated descriptions.
