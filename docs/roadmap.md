# Roadmap

## Phase 0: Static federation

- YAML-configured nodes
- OpenAI-compatible routing
- SQLite registry and ledger
- Basic policy gates
- Optional benchmarking

## Phase 1: Worker registration

- Signed worker manifests
- Authenticated registration over private mesh networking
- Worker heartbeat/drain/maintenance states
- Backend health probing

## Phase 2: Model lifecycle

- Remote model load/unload commands
- Warm model scheduling
- Idle unload policy
- CAS/hash-based model artifact identity

## Phase 3: Orchestro integration

- Use Orchestro Mesh as a backend family inside Orchestro
- Route agent inference remotely while keeping tools local
- Feed run ratings and verifier outcomes back into market scoring

## Phase 4: Advanced scheduling

- N-answer tournament mode
- Speculative cheap-first racing
- Idle/off-hours batch queues
- Heat/power/gaming-aware node availability

## Phase 5: Research modes

- LAN-only vLLM distributed serving
- Prefill/decode split
- Private Petals-like layer swarm
- KV-cache-aware placement
