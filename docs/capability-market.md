# Capability Market

The capability market is the live inventory used by the scheduler.

## Entry shape

```yaml
node_id: friend-a-4090
owner: friend-a
trust_domain: friends
status: online
hardware:
  gpus:
    - vendor: nvidia
      model: RTX 4090
      vram_gb: 24
backends:
  - id: friend-a-vllm
    kind: vllm
    base_url: http://friend-a.tailnet.example:8000/v1
models:
  - id: qwen-coder-friend-a
    backend_id: friend-a-vllm
    state: warm
    tags: [coding, fast, friends-ok]
    task_classes: [chat, coding, analysis, tool-use]
    benchmark:
      expected_decode_tps: 60.0
      expected_tool_output_tps: 32.0
      confidence: 0.35
policy:
  allowed_sensitivities: [public, friends-private]
  denied_sensitivities: [repo-private, secret-local, tool-local]
```

## Benchmarking philosophy

Benchmarks are useful but not mandatory. A market entry can advertise expected values first. Later, `orchestro-mesh benchmark` can add measured samples.

## Tool-output token rate

`expected_tool_output_tps` is intentionally separate from decode throughput. Tool-heavy tasks often include planning, structured output, JSON-ish content, and multi-step chain packaging. A model that decodes fast for prose may be worse at reliable tool-output production.
