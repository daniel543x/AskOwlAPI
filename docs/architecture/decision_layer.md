# Decision Layer (Phase 7): Jev / Laya pre-scores

Typed decision models score and filter the pipeline's intermediate
results. One Jev-compatible HTTP client serves both backends (decision
#2, 2026-10-18): a self-hosted `laya[serve]` sidecar or the Jev cloud.

## Backends and wire protocol

`POST /v1/systemone` with `{state, questions}`; questions are typed
(`choice` / `score` / `noul`) and answers carry probabilities. Batches
(`POST /v1/systemone/batch`) apply one question set over many states —
used by the search-result filter. Implemented in
`src/modules/decision/` (`JevCompatibleClient`).

### Running a Laya sidecar

```bash
podman-compose --profile laya up -d laya   # installs laya[serve]; heavy (torch)
```

Then register it as a catalog provider (admin API): provider type `jev`,
base URL `http://laya:8000` (container network) or
`http://127.0.0.1:8000` (host), optional bearer key, and a
`decision`-kind entry. The entry's `default_params` carry the settings:

```json
{
  "timeout": 30,
  "calibration": {"noul:2": 1.4, "score:5": 1.8, "default": 1.2}
}
```

## Calibration (read before trusting probabilities)

Shipped Laya checkpoints are **over-confident** (raw ECE 0.213); fitting
one temperature per `(question type, option count)` moves mean ECE to
0.081. AskOwl applies the temperatures from the entry's `calibration`
map to every answer (`softmax(logits/T)` on the returned probabilities)
before anything gates on them. **Fit your own temperatures on your own
data** (the Laya docs describe the procedure) — the defaults above are
placeholders, not fitted values. Gate on the calibrated confidence
(`answer_confidence` semantics), never on `action.act_probability`.

## Pre-scores (all OFF by default until eval'd)

| Toggle | Effect | Failure mode |
|---|---|---|
| `PRE_SCORE_SEARCH_FILTER` | noul "is this result relevant?" per search result (batched) before scraping — saves fetches | keeps all results |
| `PRE_SCORE_CHUNK_FILTER` | score 1–5 per selected chunk as a second opinion next to the cross-encoder; drops below `PRE_SCORE_CHUNK_MIN_SCORE` | keeps all chunks |
| `PRE_SCORE_QUERY_CHOICE` | choice between the rewritten and raw query | keeps the optimized query |
| `PRE_SCORE_GROUNDING_CHECK` | noul "is this answer supported by the context?" — surfaced as an SSE `grounding` event / JSON `grounding_supported` field, **never blocks** | omits the signal |

Every helper in `src/core/tools/pre_scores.py` degrades to "no opinion":
an unreachable sidecar or malformed answer never breaks a request.

The agent mode additionally gets a `pre_score` tool (yes/no questions
with calibrated probabilities) whenever a decision model resolves.

## Measuring

The eval harness runs the retrieval pre-scores when toggled:

```bash
uv run python -m src.core.ask.eval --pre-score-search-filter
uv run python -m src.core.ask.eval --pre-score-chunk-filter --decision-id <uuid>
```

A pre-score becomes a default only after beating the Phase 3 baseline on
the golden set (same gate as any retrieval change).
