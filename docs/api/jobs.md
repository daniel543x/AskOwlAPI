# Background Jobs (InBox)

`background: true` on an ask route runs the turn as a queued job and
answers `202 {job_id, kind, status, queue_position}` (PR 12). The InBox
is the user's view over the queue (PR 13): pending jobs with positions,
running/done/failed with a seen flag, live updates over SSE.

## Routes (any auth)

### `GET /jobs[?seen=]`

The caller's jobs, newest first. `?seen=false` is the unread badge.
Each job:

```json
{
  "id": "uuid", "kind": "search | research", "status":
  "queued | running | succeeded | failed | cancelled",
  "queue_position": 2, "conversation_id": "uuid | null",
  "result_summary": "string | null", "error": "string | null",
  "seen": false,
  "created_at": "...", "started_at": "...", "finished_at": "..."
}
```

### `POST /jobs/{id}/seen`

Marks the notification read; returns the job.

### `POST /jobs/{id}/cancel`

Cancels a still-queued job. Running/finished jobs answer `409`
(cooperative cancel of a running turn is not supported in v1).

### `GET /jobs/events` (SSE)

Per-user stream of `job_update` events:

```
event: job_update
data: {"job_id": "...", "kind": "research", "status": "running",
       "queue_position": null, "result_summary": null, "error": null}
```

Emitted on every transition (queued → running → succeeded/failed/
cancelled). Single-node in-process bus (decision 19); a Redis-backed
bus arrives with the multi-node topology.

## Restart behavior

A startup sweep marks jobs left `running` by a restart as failed
("Interrupted by a server restart; re-run the turn.") — the InBox never
shows a phantom running job.
