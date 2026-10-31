# Numbered citations (`wu/numbered-citations`) — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** every mode's answer carries dense `[n]` citations assigned after the fact by deterministic code, with a validator and an offline fixture eval.

**Architecture:** one pure shared module (`src/core/tools/citations.py`: registry, rewriter with a streaming carry-buffer, validator), wired into the research report assembler (registry precomputed, per-section rewrite), the search answer paths (SSE deltas + JSON + agentic + persona + no-notes fallback), a fixture-driven eval, and docs. Model output is never rejected — bad citations are left as-is and reported.

**Tech Stack:** pure Python (regex, dataclasses), FastAPI pipelines unchanged, pytest offline.

**Spec:** `docs/superpowers/specs/2026-10-31-citations-design.md` (approved 2026-10-31).

## Global Constraints

- English only; `logging` never `print`; ruff clean: `uv run ruff check src tests`.
- `uv run pytest -q` must stay green offline; DB work stays out of this branch (no schema changes).
- Pure functions: the citations module does no IO and no clock reads.
- A malformed citation never fails a run: rewrite leaves unknown/broken markers untouched; validator returns a verdict object.
- Branch `wu/numbered-citations` stacked on `wu/persona-api-batch`; PR targets `wu/persona-api-batch`; version = `0.<PR-number>.0` set after `gh pr create` reveals N (CRUSH.md).
- Pre-existing basedpyright errors across `src/core/**` are out of scope.

## Review Focus

- A source cited twice gets ONE number (first appearance); a URL cited with a different label (`[Source]` vs `[1]` vs `[doc]`) still rewrites. → Task 1 test 2.
- A link split across stream chunk boundaries (`[Sou` | `rce](url)`) never leaks raw markdown into a delta and survives to a rewritten `[n]`. → Task 1 test 5.
- Prose opening mid-list is impossible: first citation seen must be `[1]`, second distinct `[2]`, even when the model cites the 5th ranked source first. → Task 3 test 1.
- The `## Sources` section must not mark itself cited (validator scans body only). → Task 2 test 3.
- Zero-sources runs (no notes, empty ranked list) produce byte-identical output to today and a trivially-passing verdict. → Task 2 test 4, Task 3 test 4.
- JSON and SSE answers of the same run carry the same numbering. → Task 3 test 3.

---

### Task 1: The shared citations module (`src/core/tools/citations.py`)

**Files:**
- Create: `src/core/tools/citations.py`
- Create: `tests/core_tools/test_citations.py`

**Interfaces:**
- Consumes: `normalize_url` from `src/tools/urls.py` (applied to both sides of every URL comparison).
- Produces:
  - `build_registry(known_urls: Iterable[str], cited_order: Iterable[str]) -> dict[str, int]` — numbers cited (normalized) URLs by `cited_order` first, then remaining known URLs in `known_urls` order; 1-based dense.
  - `rewrite_citations(text: str, registry: dict[str, int]) -> tuple[str, dict[str, int]]` — every markdown link `[label](url)` whose normalized URL is registered becomes `[n]`; everything else (unknown links, bare URLs, prose `[3]`) passes through. Returns `(new_text, stats)` with `stats = {"rewritten": int, "left_as_is": int}`.
  - `class CitationRewriter` — `__init__(self, known_urls: list[str])`
    (priority order); assigns numbers lazily in first-appearance order
    (i.e. `build_registry` semantics emerging live): `feed(delta: str) -> str` emits only completed text (a possible partial markdown link is held back, bounded: a pending tail longer than 4096 chars with no `[` is flushed verbatim; a pending link longer than 2048 chars is flushed as-is), `flush() -> str` finishes, and `registry` / `citations_payload() -> list[dict]` expose the final mapping.
  - `CitationReport` dataclass: `cited_numbers: list[int]`, `missing_numbers: list[int]`, `unresolved_markers: list[int]`, `uncited_sources: list[int]`; and `validate_citations(body: str, registry: dict[str, int]) -> CitationReport` — scans prose `\[(\d+)\]` markers only.
  - `body_before_sources(text: str) -> str` — text before a `## Sources` heading (whole text when absent).

- [ ] **Step 1: Write the failing tests** (`tests/core_tools/test_citations.py`; plain functions, no DB)

  1. `test_registry_numbers_cited_first_then_rest` — known `[a, b, c, d]`, cited order `[c, a]` → `{c:1, a:2, b:3, d:4}` (normalized keys).
  2. `test_rewrite_rewrites_known_links_label_agnostically` — `[Source](a)` and `[doc](a)` and `[1](a)` all → `[n]` for a's number; unknown-URL link and bare URL untouched; repeated citation same number.
  3. `test_rewrite_passes_prose_numbers_through` — existing `[2]` in prose is not duplicated or altered.
  4. `test_validator_dense_report_and_unresolved` — body citing `[1]`, `[2]` with 3-entry registry → dense, `uncited_sources == [3]`; body citing `[1]`, `[4]` → `unresolved_markers == [4]`, `missing_numbers == [2]`; body before a `## Sources` heading ignores the heading's `[n]` markers (use `body_before_sources`).
  5. `test_stream_rewriter_handles_split_links` — feed `"[Sou"`, `"rce](url-a) done [x](url-b) tail"`, flush → deltas + flush contain `[1]`, `[2]`; no raw `[Source](...)` in output; partial `[Star` at end flushes verbatim.
  6. `test_stream_rewriter_first_citation_is_one_even_for_late_ranked_url` — known `[a, b, c, d, e]`, text cites `d` then `a` → outputs `[1]` for d, `[2]` for a; final registry appends `b, c, e` as 3, 4, 5.

- [ ] **Step 2: Run to verify failure**

  Run: `uv run pytest tests/core_tools/test_citations.py -q`
  Expected: FAIL — module does not exist (ImportError).

- [ ] **Step 3: Implement** `src/core/tools/citations.py` per the Interfaces block (regexes: markdown link `\[[^\]]*\]\([^)\s]+\)`, prose marker `\[(\d+)\]`; normalization at every comparison edge; the rewriter holds back at most 4096 chars of non-link tail or 2048 chars of one pending link, then flushes verbatim).

- [ ] **Step 4: Run tests**

  Run: `uv run pytest tests/core_tools -q && uv run ruff check src/core/tools tests/core_tools`
  Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/tools/citations.py tests/core_tools/test_citations.py
git commit -m "Add the shared citation registry, rewriter, and validator"
```

### Task 2: Research report assembly rewrite

**Files:**
- Modify: `src/core/research/services/synthesis.py` (`assemble_note_report_stream`, `assemble_note_report`, shared private helper)
- Modify: `src/core/research/services/pipeline.py` (validator debug-log on the final document — one call site)
- Create: `tests/core_research/test_citations.py`

**Interfaces:**
- Consumes: Task 1 module (`build_registry`, `rewrite_citations`, `validate_citations`, `body_before_sources`).
- Produces: both assembly functions emit the SAME registry semantics: intro/section parts built as today, then each part rewritten, `## Sources` appended from the registry as `- [n] url` (replacing the `- [Source](url)` list); section "Citations:" lines become `Citations: [1] [3]`; registry = first appearance in the assembled body of any `note.sources_used` URL, then consulted-uncited appended in note order. `generate_report`/`stream_report` (no-notes fallback) pass through unchanged when they have no sources (empty registry → no-op).

- [ ] **Step 1: Write the failing tests** (`tests/core_research/test_citations.py`; reuse the Persona/note construction patterns from `tests/core_research/test_persona_mode.py` — plain objects, no DB)

  1. `test_assembly_numbers_by_first_appearance_and_lists_sources` — two notes: note B's URL cited first in the intro-adjacent section; assert dense `[1]`, `[2]` in section lines and body, `## Sources` lists `- [1] urlB`, `- [2] urlA`.
  2. `test_assembly_appends_consulted_but_uncited` — note with `sources_used` URL never appearing inline → listed last in Sources, validator `uncited_sources` names it, cited prefix stays dense.
  3. `test_validator_scans_body_not_sources_section` — a body citing `[1]` only, Sources listing `[1]`, `[2]` → verdict from the body still reports `uncited_sources == [2]`.
  4. `test_no_sources_report_is_unchanged` — notes with empty `sources_used` → output byte-identical to the pre-change assembly for the same notes/plan (no `## Sources`, no markers).
  5. `test_stream_and_sync_assembly_match` — same notes/plan through both functions → identical final strings.

- [ ] **Step 2: Run to verify failure**

  Run: `uv run pytest tests/core_research/test_citations.py -q`
  Expected: FAIL (Sources still `- [Source](url)`, no numbering).

- [ ] **Step 3: Implement** — extract the shared part-builder, compute the registry from the built body, rewrite parts, append the Sources section from the registry; debug-log `validate_citations(body, registry)` in the pipeline after assembly.

- [ ] **Step 4: Run tests**

  Run: `uv run pytest tests/core_research -q && uv run pytest -q`
  Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/research/services/synthesis.py src/core/research/services/pipeline.py tests/core_research/test_citations.py
git commit -m "Number research report citations deterministically at assembly"
```

### Task 3: Search, agentic, persona, and fallback answer paths

**Files:**
- Modify: `src/core/ask/services/search.py` (SSE deltas via `CitationRewriter`; `citations` event after the last `message` delta; JSON answer rewritten + `"sources"` payload)
- Modify: `src/core/ask/agentic.py` (answer rewrite with the toolbox's ranked sources)
- Modify: `src/core/research/services/agent.py` (persona quick loop's final `SubmitOutcome` answer rewritten with `toolbox.ranked_sources`)
- Modify: `src/core/research/services/pipeline.py` (no-notes fallback: registry from gathered context URLs; no-op when empty)
- Create: `tests/core_ask/test_citations.py`

**Interfaces:**
- Consumes: Task 1 module; `toolbox.ranked_sources` (existing) as the known-URL priority list in every mode.
- Produces:
  - SSE search: `message` deltas carry `[n]`; a new terminal-direction event `citations` with payload `{"citations": [{"n": 1, "url": "..."}, ...]}` is yielded after the last `message` delta and before `done`/`error`; `_save_turn` persists the rewritten answer.
  - JSON search: `{"answer": ..., "sources": [{"n", "url"}, ...]}` (additive).
  - Persona quick + agentic + fallback: final answer text rewritten; no protocol/shape change beyond the text itself.
  - In every mode: zero known sources → byte-identical output to today.

- [ ] **Step 1: Write the failing tests** (`tests/core_ask/test_citations.py`; harness copied from `tests/core_ask/test_search_pipeline.py` — fakes + `collect_sse`)

  1. `test_sse_numbers_dense_by_first_citation` — ranked sources `[a, b, c, d, e]`, scripted answer citing `d` then `a` → deltas show `[1]` (for d) then `[2]` (for a); a `citations` event arrives after the last `message` with the full mapping (`d:1, a:2, b:3, c:4, e:5`); `done` still terminal.
  2. `test_sse_split_link_never_leaks` — a scripted model that streams the answer in two chunks splitting a link mid-marker → no delta contains raw `[Source](`.
  3. `test_json_sources_match_sse_numbering` — same inputs via JSON adapter → `answer` numbering identical to the SSE run's, `sources` equal to the SSE `citations` payload.
  4. `test_no_sources_answer_is_unchanged` — empty ranked list → answer identical to pre-change behavior, no `citations` event.
  5. `test_persona_and_agentic_answers_rewritten` (in `tests/core_research/test_persona_agents.py` + `tests/core_ask/test_agentic.py`) — scripted answers citing ranked URLs → final `message` carries `[n]`.

- [ ] **Step 2: Run to verify failure**

  Run: `uv run pytest tests/core_ask/test_citations.py -q`
  Expected: FAIL (no citations event, links unrewritten).

- [ ] **Step 3: Implement** per Interfaces. The `citations` event is yielded by the SSE adapter (not the pipeline generator) right before `done` so it also covers the agentic path; keep `_save_turn` on the rewritten text.

- [ ] **Step 4: Run tests**

  Run: `uv run pytest tests/core_ask tests/core_research -q && uv run pytest -q`
  Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core tests/core_ask tests/core_research
git commit -m "Rewrite answers to numbered citations across search, persona, and agentic modes"
```

### Task 4: Fixture eval, docs, bookkeeping

**Files:**
- Create: `tests/fixtures/citations/` (input/expected JSON pairs) + `tests/core_tools/test_citations_fixture_eval.py`
- Modify: `docs/api/ask_search_sse.md` (changelog: `citations` event, JSON `sources` field), `docs/api/ask_research_sse.md` (report body carries `[n]`; `## Sources` is the numbered registry)
- Modify: `PLAN.md` (tick Phase 12 PR 3; note the stacked branch)
- Version bump commit per CRUSH.md after `gh pr create` reveals N

**Interfaces:** none (eval + docs).

- [ ] **Step 1: Fixtures** — ≥8 pairs covering: repeated URLs, label variants, unknown-URL links, bare URLs, uncited consulted sources, malformed markers (`[3]` with empty registry, `[0]`, `[99]`), split-link replay, empty-sources no-op. Format per pair: `{"text": ..., "known": [...], "expected_text": ..., "expected_registry": {...}, "expected_verdict": {...}}`.
- [ ] **Step 2: Eval test** — `test_citations_fixture_eval` iterates the directory, runs rewrite + validate (and assembly where the fixture includes notes), asserts equality with expected values, and reports the `uncited_sources` metric per fixture in the assertion message (never fails on it).
- [ ] **Step 3: Docs** — changelog entries as above.
- [ ] **Step 4: PLAN.md** — tick Phase 12 PR 3 with the branch name.
- [ ] **Step 5: Full verification + commit**

  Run: `uv run pytest -q && uv run ruff check src tests`
  Expected: PASS.

```bash
git add tests/fixtures/citations tests/core_tools/test_citations_fixture_eval.py docs/api PLAN.md
git commit -m "Add the citations fixture eval and document the protocol additions"
```
