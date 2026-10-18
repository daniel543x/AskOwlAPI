# Error Reference

## HTTP status codes

| Code | Meaning | Where |
|---|---|---|
| 204 | Success, no body | all `DELETE` endpoints (personas, api-keys, history, users, model catalog) |
| 400 | Malformed request body | user creation with duplicate email/nickname, invalid model-catalog payload |
| 401 | Missing/invalid credentials (bad JWT, unknown/revoked/expired API key, wrong login) | all authenticated routes |
| 403 | Authenticated but not allowed (missing role, missing API-key scope, blocked user) | elevated modes, admin routes |
| 404 | Not found **or owned by someone else** | personas, history, conversations, api-keys, unknown `conversation_id` in JSON mode |
| 409 | Conflicting state (duplicate provider name, admin blocking own account) | model catalog, admin user management |
| 422 | Validation failed (unknown scope, no tool-capable model) | api-key creation, model resolution |
| 429 | API-key rate limit exceeded (`Retry-After` header) | X-API-Key traffic |
| 500 | Pipeline failure | ask/research JSON mode |
| 502 | Model violated the tool-call contract (must support tool calling) | answer/report synthesis |
| 503 | No model resolvable for the requested kind (registry empty / provider down) | model resolution before the pipeline runs |

## SSE `error` events

Once an SSE stream has started, errors are terminal `error` events
(never mid-stream HTTP errors):

```
event: error
id: N
data: {"error": "description"}
```

Common cases: no search results, model responded without the required tool
call, answer/report generation failure, conversation not found (when the
`conversation_id` was requested with streaming).

Terminal events: `done` and `error`; treat both as stream end.
