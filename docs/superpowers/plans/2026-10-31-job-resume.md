# Job auto-resume (`wu/job-resume`) — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (native, inline).

**Goal:** a restart re-enqueues interrupted background runs; research/deep runs continue from the last research note — covered gather legs are skipped, synthesis covers old + new notes, nothing duplicates.

**Architecture:** the research pipeline gets an optional progress hook (plan tasks + note ids written back into `Job.payload`, JSON only — no schema changes) and skips gather legs whose task already has a note in the (pre-seeded) note store. A new `src/modules/jobs/resume.py` rebuilds saved runners from the job row for kinds `search` and `research`, reusing the underlying factories with the engine's own session, and the startup sweep re-enqueues instead of failing (retry-capped by `JOB_MAX_RETRIES`).

**Tech Stack:** asyncio queue rework, SQLModel factories, pytest offline + integration.

**Spec:** approved design (2026-10-31, conversation): option 1 — note-informed resume, no duplication; re-enqueue replays whole turns only for search-kind jobs.

## Global Constraints

- User runs whose retries are exhausted keep today's sweep behavior (failed + message), re-enqueue shows up as `queued` again.
- Byte-identical behavior when no orphans exist, and for foreground asks (untouched paths).
- The model/user/tool resolution for rebuilt runners mirrors what the enqueue used (user lookup by job.user_id for the roles; profile default for the chat model; source_types default).
- Commands: `uv run pytest`, `uv run pytest -m integration`, `uv run ruff check src tests`.
- Based on branch `wu/soft-delete`; version `0.<PR>.0` (CRUSH.md).

## Review Focus

- An interrupted research run resumes with ZERO new notes taken for a covered task and synthesis still cites the pre-restart notes. → Task 2 test 2.
- A search-kind job replays whole; no payload "research" key required. → Task 2 test 1.
- Orphans exceeding the retry cap still fail with today's message. → Task 2 test 3.
- The rebuilt research runner must not re-plan (persisted plan is authoritative on resume). → Task 2 test 2.
- No orphan → startup and jobs byte-identical to today. → Task 3 integration test 1.

---

### Task 1: Pipeline progress hook, NOTE-prefilled store, and skip logic

**Files:**
- Modify: `src/modules/notes/store.py` (`NoteStore.__init__(..., seeded: list[ResearchNote] | None = None)` — seeded rows join `self.notes` so `ordered_notes` returns old+new in sequence order)
- Modify: `src/core/research/services/pipeline.py`: `research_json_adapter` gains `progress: Callable[[dict], Awaitable[None]] | None = None`; called after the planner (`{"planned_tasks": [[idx, task], ...]}`) and after each successful note write (`{"note_ids": [...expanded]}`); the gather loop skips a subquestion when a seeded note with that task exists (`done_tasks = {n.task for n in note_store.notes}` computed once; planner's `subquestions` filtered against it — the event stream still reports the plan with a `"resumed": true` flag when skipping)
- Test: `tests/core_research/test_resume_progress.py` (offline; fakes)

- [ ] Steps: RED for (a) seeded store ordering, (b) progress hook payload shape, (c) skip of covered tasks with synthesis over both sets → implement → suite → commit `Persist research progress and skip covered legs on resume`.

### Task 2: Runner rebuild + re-enqueue sweep

**Files:**
- Create: `src/modules/jobs/resume.py`:
  - `async def build_runner(job, session) -> Runner | None` — `search`: full-turn replay of `search_json_adapter(query, deps..., session, user_id, conversation_id)`; `research`: rebuild deps via `SearchProviderFactory.create_provider(session, None)`, `ScraperFactory.create(session)`, `_resolve_ranker_model` + `RankerFactory.from_resolved`, embedder via the embedding repository (None on failure — same degrade), chat model via `ModelRepository.resolve_model` with the job's user/roles/profile defaults, persona via `PersonaRepository.get_for_user` when `payload["persona_id"]` present, `note_store` seeded from `repo.get_by_ids(payload["research"]["note_ids"])`, progress = writer saving payload through its own engine session.
  - `async def requeue_orphans(queue, session_factory) -> int` — selects `running` rows; `retries >= JOB_MAX_RETRIES` → keep the old fail-sweep; else `status=queued`, keep payload/retries+1, `queue.requeue(job, runner)`.
- Modify: `src/modules/jobs/queue.py` (public `requeue(job, runner)` — append + wake, no status rewrite beyond what the requeue sets)
- Modify: `src/modules/jobs/repository.py` (expose an iteration primitive the requeue uses; keep `sweep_orphans` for the retry-exhausted path only — signature may gain `only_exhausted=True`)
- Modify: `src/app.py` lifespan: replace the fail-sweep call with `requeue_orphans` + log; keep the blob sweep untouched
- Test: `tests/modules/jobs/test_resume.py` (integration when it touches the DB; queue-level fakes for the registry)

- [ ] Steps: RED (requeue method, requeue_orphans semantics: one requeued job runs to success with the fake runner and the retry cap path) → implement → suites → commit `Re-enqueue interrupted background runs from their payloads`.

### Task 3: End-to-end integration + bookkeeping

- Test: `tests/modules/jobs/test_resume_e2e.py` (integration): script a research job row with payload progress (planned tasks, note_ids) and a REAL note row; intake JobQueue with the real resume registry but STUB model/tool deps (monkeypatched factories); assert the resumed runner completes, the covered leg produced no new note, synthesis includes the seeded note.
- Tick the PLAN.md backlog line (`Job auto-resume`).

- [ ] Steps: RED → green → full suites → commit, PR + version `0.<PR>.0`.
