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

## Repo status

This is an implementation scaffold, not a hardened distributed platform yet. The high-value next pass is to wire this directly into the existing Orchestro backend registry and add authenticated worker-to-gateway registration over WireGuard/Tailscale/Headscale.
