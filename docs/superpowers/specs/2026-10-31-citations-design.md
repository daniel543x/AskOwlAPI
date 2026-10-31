# Numbered citations (`wu/citations`) — design

Date: 2026-10-31 · Phase 12 PR 3 · Owner: daniel · Status: awaiting approval

## Intent and why

Today every mode tells the model to "cite sources inline as `[Source](URL)`"
and nothing checks the result. The webapp wants stable, numbered `[n]`
citations it can render and verify. Numbers are assigned **after the fact**
by deterministic code, not by the model — the model keeps citing URLs the
way it already does, and the assembler turns those into a dense `[1..N]`
sequence. The owner picked this explicitly: it avoids gaps and missing
numbers by construction, and the model cannot violate the numbering.

## Goals

- `[n]` citations in **all modes**: quick search answers (`CitedAnswer`)
  and research/deep reports (the Note assembler).
- A dense sequence: prose cites `[1..K]`, no gaps, no orphans; every `[n]`
  maps to a listed source.
- A `## Sources` section (research) / machine-readable `sources` list
  (search) that clients can resolve `[n]` against.
- A validator with a machine-readable verdict (dense/unresolved/uncited)
  and an offline fixture eval pinning it.

## Non-goals

- No prompt changes asking the model to number anything.
- No repair round for bad citations (bad markers are left as-is and
  reported, never retried — the agent loop's budget is not spent here).
- No dedup/amnesty logic across turns; numbering is per answer/report.
- No UI work.

## Anti-goals (failure modes to avoid)

- Renumbering that changes between the SSE stream and the JSON answer of
  the same run (both paths must share one registry).
- Sources listed with numbers that prose can never produce (holes).

## Constraints

- Async end-to-end; pure functions for extract/rewrite/validate (no IO).
- English only; `logging` never `print`; ruff + offline pytest gates.
- Additive protocol changes only; existing clients keep working.

## Approaches considered

1. **Assembler assigns (chosen)** — code collects the URLs the model
   cited, numbers them by first appearance, rewrites the text, and builds
   the Sources list. Cannot be violated by the model; no prompt changes;
   trivially testable. Tradeoff: prose says `[3]` without knowing why
   (acceptable; clients resolve via the Sources list / sources payload).
2. **Model assigns** — the model gets a pre-numbered source list and cites
   `[n]`; validator triggers a repair round. More natural prose, but adds
   a repair path, prompt surface, and failure modes the loop just learned
   to survive. Rejected for this PR.

## Decided

- **Shared module** `src/core/tools/citations.py` (pure, offline):
  - `build_registry(urls: Iterable[str], priority: Iterable[str] | None)`
    → ordered `dict[url, int]`. Research priority: inline appearance;
    search priority: `ranked_sources` order.
  - `rewrite_citations(text, registry)` → rewritten text + stats. A
    markdown link is a citation when its URL is in the registry
    (label-agnostic: `[Source](url)`, `[1](url)`, `[anything](url)`);
    it becomes `[n]`. Links to unknown URLs and bare URLs pass through
    untouched. Body-marker regex `\[(\d+)\]` (already-numeric labels)
    are left alone.
  - URL comparison runs both sides through the existing
    `normalize_url` in `src/tools/urls.py` (single copy; the eval copy
    stays out of prod code).
  - `validate_citations(text, registry)` → `CitationReport` dataclass:
    `cited_numbers` (dense?), `missing_numbers`, `unresolved_markers`
    (prose `[n]` with n > N or n < 1), `uncited_sources` (registry URLs
    never cited in prose). Not an exception — a verdict object.
- **Density is the invariant in every mode** (the owner's stated goal:
  no gaps, no missing numbers). Search numbers are dense-by-first-
  citation, NOT ranked positions: the registry orders cited URLs by
  first appearance, then appends never-cited ranked sources (so the
  list is complete and the cited prefix `[1..K]` is dense). To keep
  clients able to resolve numbers, the SSE search path gains an
  additive `citations` event (`{"citations": [{"n": 1, "url": ...},
  ...]}`) emitted right after the last `message` delta (the registry
  only finalizes with the text, so it cannot precede the deltas), and
  the JSON response gains the same list as `"sources"`. The existing `sources`
  event is unchanged (it stays the ranked retrieval view).
- **Search rewrite is live, not buffered**: the delta stream passes
  through a carry-buffer rewriter that holds back only a possible
  partial markdown link (bounded tail), rewrites complete links to
  `[n]`, and flushes at answer end. The persisted answer and the JSON
  answer use the same fully rewritten string — no stream/JSON
  divergence. Grounding checks run pre-rewrite (they match URLs, not
  markers).
- **Scope is genuinely all modes** (owner: "apply to all, cost
  acceptable"): the linear CitedAnswer path, the agentic search answer
  path, the persona quick loop's SubmitOutcome answer, and the
  research/deep report — plus the no-notes research fallback
  (`generate_report`/`stream_report`), whose registry comes from the
  gathered context URLs; with zero sources the rewrite is a no-op and
  the validator trivially passes (graceful degrade, never an error).
- **Research/deep** (`assemble_note_report_stream` / `assemble_note_report`):
  1. Assemble intro + sections as today (section "Citations:" lines keep
     their links for now).
  2. Registry = first appearance of any note `sources_used` URL in the
     assembled body, then consulted-but-uncited URLs appended in note
     order (still listed, never holes in the cited prefix).
  3. Rewrite pass over the body (section citation lines become `[n]`
     lists; inline links in note content become `[n]`).
  4. `## Sources` built from the registry: `- [n] url`.
  5. Validator runs on the final document; verdict logged (debug) and
     counted in the fixture eval.
- **Research assembly stays incremental**: the registry is computed up
  front (notes and plan are known before the first yield), each yield
  is an atomic section/intro unit rewritten on its way out, and the
  `## Sources` section is appended last from the registry —
  time-to-first-byte is preserved, no whole-report buffering.
- **Fixtures + eval**: a fixture directory of messy model outputs (repeated
  URLs, unknown URLs, bare URLs, uncited consulted sources, malformed
  markers, empty sources) with expected rewritten text + registry +
  validator verdicts; the eval asserts rewrite determinism and validator
  verdicts, and reports `uncited_sources` as a metric (never a failure).
- **Validator wiring**: verdicts are logged and asserted in tests; the
  validator always scans the BODY only — for reports, everything before
  the `## Sources` heading — so Sources lines cannot mark themselves
  cited. No user-facing error path in this PR (a report is never
  rejected over citations).

## Left to the builder

- Exact regexes for markdown-link extraction and prose-marker scanning.
- Fixture file format and count (cover the listed mess classes).
- Whether the validator verdict rides the JSON search response (default:
  it does not — log only).
