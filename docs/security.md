# Security Model

Orchestro Mesh starts with conservative private-federation defaults.

## Core rule

Remote nodes may provide tokens. They do not get execution authority.

## Sensitivity classes

| Class | Meaning | Default route |
|---|---|---|
| `public` | Non-sensitive prompts | Any allowed node |
| `friends-private` | Trusted friend-pool data | Trusted friend nodes |
| `repo-private` | Private repo/code context | Explicitly trusted local/private nodes only |
| `secret-local` | Credentials, tokens, keys, medical/legal/financial | Local only |
| `tool-local` | Tool execution and mutation authority | Local only |

## Node policy

Nodes declare:

- allowed users
- allowed sensitivities
- denied sensitivities
- max concurrent jobs
- max context/output limits
- remote tool authority stance
- tags

## Recommended network posture

- Put the mesh on Tailscale, Headscale, or WireGuard.
- Expose only worker-agent ports to the mesh.
- Keep raw llama.cpp/Ollama/vLLM ports bound to localhost on each worker.
- Do not expose `/tools`, MCP bridges, shell execution, or repo-write APIs to remote nodes.

## Redaction

The scaffold includes basic pattern redaction for obvious secrets. It is intentionally conservative and incomplete. Treat it as a tripwire, not as a full DLP system.
