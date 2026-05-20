# Orchestro Mesh

Private trusted inference mesh and capability market, with orchestrater (Orchestro)

Orchestro Mesh gives a small trusted group one local OpenAI-compatible endpoint while coordinating multiple private GPU/CPU nodes behind it. The project is intentionally **request-level federation first**, not premature distributed tensor parallelism over residential internet.

## Design thesis

- Remote machines provide inference horsepower, not execution authority.
- Tool execution, file mutation, git writes, Kubernetes access, secrets, and repo authority stay local unless explicitly allowed.
- Every node advertises a capability-market entry: hardware, backends, loaded models, task tags, trust policy, current load, and optional benchmark measurements.
- The scheduler routes jobs by task fit, trust fit, warm model state, queue depth, and observed/expected token throughput.
- Benchmarking is supported but optional; market entries can start with expected `tok/s` and then improve as real samples are collected.

## Current scaffold

This initial repo includes:

- FastAPI gateway exposing `/v1/chat/completions` and mesh management routes.
- Worker app exposing inventory and proxy inference routes.
- Capability market models for nodes, backends, models, policies, and benchmark profiles.
- Scheduler with policy gates and scoring.
- SQLite node/job/usage ledger.
- CLI for node listing, route simulation, and optional benchmarking.
- Redaction helpers for obvious secrets.
- Example YAML configuration.
- Pytest coverage for policy, scheduler, redaction, and storage.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
cp config/mesh.example.yaml mesh.yaml
pytest
orchestro-mesh nodes --config mesh.yaml
orchestro-mesh route "write a small Rust parser" --config mesh.yaml --task coding
uvicorn orchestro_mesh.gateway:app --host 127.0.0.1 --port 8765
```

Point OpenAI-compatible clients at:

```text
http://127.0.0.1:8765/v1
```

## Capability market benchmark fields

Each model advertises a `benchmark` profile:

```yaml
benchmark:
  expected_prompt_tps: 500.0
  expected_decode_tps: 45.0
  expected_end_to_end_tps: 38.0
  expected_tool_output_tps: 22.0
  confidence: 0.4
  samples: []
```

`expected_tool_output_tps` is the scheduler's estimate for tool-planning / tool-output-heavy tasks. Real benchmark samples can later populate `samples` with `prompt_tps`, `decode_tps`, `end_to_end_tps`, and `tool_roundtrip_tps`.

## Safety defaults

The default policy blocks:

- `secret-local` sensitivity from remote nodes
- `tool-local` sensitivity from remote nodes
- remote tool authority
- tool-use jobs on non-local trust domains

That means a remote GPU can propose text, patches, summaries, plans, and model judgments, but it cannot mutate your machine.

Inbound `tools` / `tool_choice` in an OpenAI payload are detected and upgraded to `tool-use` automatically — clients do not need to set a custom `task_class` field to get the local-only guarantee.

## Auth

Gateway and worker endpoints accept a bearer token. Configure one of:

- `mesh_token` (single cluster-wide secret; per-user identity via `X-Orchestro-Requester` header)
- `api_tokens: {token: requester}` (per-user tokens; token determines the requester)
- `worker_token` (separate shared secret for worker `/inventory` and `/infer`)

Environment overrides: `ORCHESTRO_MESH_TOKEN`, `ORCHESTRO_MESH_WORKER_TOKEN`, `ORCHESTRO_MESH_REDACTION_MODE` (`off`/`log`/`block`), `ORCHESTRO_MESH_STORE_PATH`.

If no token is configured the gateway runs open and logs a warning — intended only for local development.

## Liveness and retry

- `node_ttl_seconds` (default 300): nodes whose `last_seen` is older than the TTL are filtered out of routing. The `local_node_id` is exempt.
- `POST /mesh/heartbeat` accepts `{node_id, status?, current_jobs?, queue_depth?}` and refreshes `last_seen`. Workers ship a built-in pusher (`orchestro_mesh.worker_heartbeat.HeartbeatLoop`) that fires on `ORCHESTRO_MESH_HEARTBEAT_INTERVAL_S` (default 30s) when `ORCHESTRO_MESH_GATEWAY_URL` is set.
- `POST /mesh/probe/{node_id}` hits each backend's `/v1/models` endpoint and updates `ModelState` (present → `warm`, missing → `absent`).
- On a transient backend error (5xx, 429, transport/timeout) the gateway falls back to the next-best candidate. The response includes `orchestro_mesh.attempts`.

## Accounting

- Token counts use `tiktoken` if installed, otherwise a `len(text) // 4` fallback. Counts feed both the context-window gate and the ledger.
- Each completed job writes a `UsageLedgerEntry` with `credit_cost = (input + output) / 1000 * credits_per_1k_tokens` (default `1.0`).
- `quota_credits: {requester: max_credits}` enforces a soft cap — requests beyond the cap return `429`.
- `rate_limit_per_minute: {requester: int}` and `rate_limit_default_per_minute` enforce in-process per-minute caps; the response includes a `Retry-After` header. Not safe across multiple gateway replicas — see "Known limits" below.
- `GET /mesh/usage` exposes per-requester totals.

## Auto-probe

Set `probe_interval_s` (e.g. `60`) and the gateway will periodically call `refresh_node_model_states` on every known node, promoting newly-loaded models to `warm` and demoting absent ones to `absent`. Backends that fail to respond are left alone and logged.

## Feedback loop

Each chat completion response carries `orchestro_mesh.request_id` (streaming responses surface the same value via the `X-Orchestro-Mesh-Request-Id` header). Clients can post verifier outcomes back:

```
POST /v1/feedback/{request_id}
{ "rating": 0.0..1.0, "verifier": "python-syntax", "notes": "..." }
```

Ratings are stored in the `feedback` table. The scheduler consults the per-`(node, model)` average and nudges the route score by `±40` at the extremes — so a model that consistently fails verifiers will slide down the ranking even if its raw throughput is high.

The companion Orchestro repo (`orchestro/src/orchestro/backends/mesh.py` + `mesh_feedback.py`) provides a `MeshBackend` that talks to this gateway and emits feedback automatically from the verifier path. Rating math: passing verifier with no warnings → 1.0, each warning deducts 0.1 (floored at 0.6), failing verifier → 0.0. Set `ORCHESTRO_MESH_VERIFIER_WEIGHTS` (JSON object) to weight verifiers differently — e.g. `{"tests": 4, "lint": 1}` makes a failed test count four times as much as a failed lint.

## Dashboard

`GET /dashboard` serves a server-rendered HTML status page: nodes (status, in-flight jobs, heartbeat age, stale-flag), models advertised per node, recent jobs, per-requester usage vs quota, and recent feedback ratings. Auto-refreshes every 5s. Auth accepts either `Authorization: Bearer <token>` or HTTP Basic (any username, token as password) so browsers can prompt without bearer wrangling. In `mesh_token` mode the Basic username becomes the requester identity.

## Cross-repo CI

`.github/workflows/e2e.yml` checks out both `snapetech/orchestro-mesh` and `snapetech/orchestro`, installs them, and runs `tests/e2e/test_cross_repo.py` — which boots a real uvicorn gateway on a free port and drives it with a real `MeshBackend`. This catches contract drift between the two repos.

## Worker robustness

- Drain mode: `POST /drain` flips status to `DRAINING`, the lifespan shutdown waits up to `shutdown_grace_s` for in-flight jobs to complete; `POST /resume` clears it.
- Local in-flight counter is shipped in every heartbeat and exposed via `/health`.
- Heartbeat applies capped exponential backoff on failure.
- `/infer` re-runs the policy check defensively, even when the gateway has already approved the request.

## Known limits

- The rate limiter and `current_jobs` counter are per-process. A multi-replica deployment needs an external backing store (Redis, Postgres) before either is trustworthy.
- The ledger is SQLite-backed; concurrent writers depend on WAL + `BEGIN IMMEDIATE` semantics. Fine for a single gateway, not for horizontal scale.

## Repo status

This is an implementation scaffold, not a hardened distributed platform yet. The high-value next pass is to wire this directly into the existing Orchestro backend registry and add authenticated worker-to-gateway registration over WireGuard/Tailscale/Headscale.
