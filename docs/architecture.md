# Architecture

Orchestro Mesh is a private inference federation control plane.

## Non-goals for v0

- No cross-internet tensor sharding.
- No remote shell execution.
- No remote filesystem authority.
- No raw model-server exposure outside the worker host.
- No public-swarm semantics.

## Runtime planes

1. **Gateway**: the local endpoint clients use. It exposes OpenAI-compatible routes and mesh control routes.
2. **Governor/Scheduler**: scores candidates from the capability market and applies policy before routing.
3. **Worker**: runs next to local inference backends, advertises capabilities, and proxies inference.
4. **Backend**: llama.cpp, Ollama, vLLM, TabbyAPI, SGLang, MLX, or another OpenAI-compatible runtime.
5. **Ledger**: SQLite persistence for nodes, jobs, and usage accounting.

## Request flow

```text
client -> gateway /v1/chat/completions -> scheduler -> selected node/model/backend -> backend response -> client
```

## Capability market

Every worker advertises:

- node owner and trust domain
- GPU/CPU/RAM inventory
- backend endpoints
- model capabilities
- task-class tags
- warm/cold state
- queue depth and current jobs
- expected token throughput
- optional benchmark samples
- node policy

## Model benchmark fields

The scheduler uses these values:

- `expected_prompt_tps`
- `expected_decode_tps`
- `expected_end_to_end_tps`
- `expected_tool_output_tps`
- `confidence`
- observed `samples`

Observed samples override expected values when present.
