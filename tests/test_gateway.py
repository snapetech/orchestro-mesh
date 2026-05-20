from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from orchestro_mesh.config import MeshConfig
from orchestro_mesh.gateway import create_gateway_app
from orchestro_mesh.models import (
    BackendEndpoint,
    BackendKind,
    ModelCapability,
    ModelState,
    NodeInventory,
    NodePolicy,
    Sensitivity,
    TaskClass,
)
from orchestro_mesh.store import MeshStore


def _local_node() -> NodeInventory:
    return NodeInventory(
        node_id="local",
        owner="keith",
        trust_domain="local",
        backends=[BackendEndpoint(id="mock", kind=BackendKind.MOCK, base_url="http://localhost/v1")],
        models=[
            ModelCapability(
                id="local-mock",
                backend_id="mock",
                state=ModelState.WARM,
                task_classes=[TaskClass.CHAT, TaskClass.CODING, TaskClass.TOOL_USE],
            )
        ],
        policy=NodePolicy(
            allowed_users=[],
            allowed_sensitivities=[Sensitivity.PUBLIC, Sensitivity.SECRET_LOCAL, Sensitivity.TOOL_LOCAL],
            denied_sensitivities=[],
            max_concurrent_jobs=4,
            allow_remote_tool_authority=True,
        ),
    )


def _friend_node() -> NodeInventory:
    return NodeInventory(
        node_id="friend",
        owner="friend-a",
        trust_domain="friends",
        backends=[BackendEndpoint(id="fmock", kind=BackendKind.MOCK, base_url="http://friend/v1")],
        models=[
            ModelCapability(
                id="friend-mock",
                backend_id="fmock",
                state=ModelState.WARM,
                task_classes=[TaskClass.CHAT, TaskClass.CODING, TaskClass.TOOL_USE],
            )
        ],
        policy=NodePolicy(
            allowed_users=[],
            allowed_sensitivities=[Sensitivity.PUBLIC],
            denied_sensitivities=[Sensitivity.SECRET_LOCAL, Sensitivity.TOOL_LOCAL],
            max_concurrent_jobs=4,
            allow_remote_tool_authority=False,
        ),
    )


@pytest.fixture
def gateway(tmp_path):
    config = MeshConfig(
        local_node_id="local",
        store_path=str(tmp_path / "mesh.db"),
        nodes=[_local_node(), _friend_node()],
        api_tokens={"tok-keith": "keith", "tok-anon": "anonymous"},
        redaction_mode="log",
    )
    app = create_gateway_app(config)
    client = TestClient(app)
    return client, config


def test_chat_completions_requires_auth(gateway):
    client, _ = gateway
    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 401


def test_chat_completions_returns_mock_response_and_writes_ledger(gateway, tmp_path):
    client, config = gateway
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer tok-keith"},
        json={"messages": [{"role": "user", "content": "hello"}], "temperature": 0.2},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "[mock-response]"

    assert body["orchestro_mesh"]["credit_cost"] >= 0.0
    store = MeshStore(config.store_path)
    totals = store.ledger_totals()
    assert totals.get("keith", 0.0) > 0.0
    with store.connect() as conn:
        rows = conn.execute("SELECT status, output_tokens FROM jobs").fetchall()
        ledger = conn.execute("SELECT requester, credit_cost FROM usage_ledger").fetchall()
    assert rows and rows[0]["status"] == "completed"
    assert rows[0]["output_tokens"] >= 1
    assert ledger and ledger[0]["requester"] == "keith"
    assert ledger[0]["credit_cost"] > 0.0


def test_node_concurrency_returns_to_zero_after_request(gateway):
    client, config = gateway
    client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer tok-keith"},
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    store = MeshStore(config.store_path)
    node = store.get_node("local")
    assert node is not None
    assert node.current_jobs == 0


def test_tools_payload_forces_local_route(gateway):
    client, config = gateway
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer tok-keith"},
        json={
            "messages": [{"role": "user", "content": "use a tool"}],
            "tools": [{"type": "function", "function": {"name": "noop", "parameters": {}}}],
        },
    )
    assert response.status_code == 200
    store = MeshStore(config.store_path)
    with store.connect() as conn:
        rows = conn.execute("SELECT node_id, task_class FROM jobs").fetchall()
    assert rows and rows[0]["node_id"] == "local"
    assert rows[0]["task_class"] == TaskClass.TOOL_USE.value


def test_redaction_block_mode_returns_400(tmp_path):
    config = MeshConfig(
        local_node_id="local",
        store_path=str(tmp_path / "mesh.db"),
        nodes=[_local_node()],
        api_tokens={"tok-keith": "keith"},
        redaction_mode="block",
    )
    client = TestClient(create_gateway_app(config))
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer tok-keith"},
        json={"messages": [{"role": "user", "content": "api_key=supersecret123"}]},
    )
    assert response.status_code == 400
    assert "findings" in response.json()["detail"]


def test_redaction_log_mode_does_not_block(gateway):
    client, _ = gateway
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer tok-keith"},
        json={"messages": [{"role": "user", "content": "api_key=supersecret123"}]},
    )
    assert response.status_code == 200


def test_mesh_nodes_requires_auth(gateway):
    client, _ = gateway
    assert client.get("/mesh/nodes").status_code == 401
    response = client.get("/mesh/nodes", headers={"Authorization": "Bearer tok-keith"})
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_retry_falls_back_to_next_candidate(tmp_path, monkeypatch):
    import httpx

    from orchestro_mesh import openai_client

    flaky = NodeInventory(
        node_id="flaky",
        owner="keith",
        trust_domain="local",
        backends=[BackendEndpoint(id="bad", kind=BackendKind.MOCK, base_url="http://flaky/v1")],
        models=[
            ModelCapability(
                id="flaky-m",
                backend_id="bad",
                state=ModelState.WARM,
                task_classes=[TaskClass.CHAT],
                benchmark={"expected_decode_tps": 200.0},
            )
        ],
        policy=NodePolicy(
            allowed_users=[],
            allowed_sensitivities=[Sensitivity.PUBLIC],
            denied_sensitivities=[],
            max_concurrent_jobs=4,
            allow_remote_tool_authority=True,
        ),
    )
    config = MeshConfig(
        local_node_id=None,
        store_path=str(tmp_path / "mesh.db"),
        nodes=[flaky, _local_node()],
        api_tokens={"tok-keith": "keith"},
    )

    original = openai_client.OpenAICompatClient.chat_completions

    async def maybe_fail(self, request, model_id, raw_payload=None):
        if self.backend.id == "bad":
            raise httpx.ConnectError("unreachable")
        return await original(self, request, model_id, raw_payload=raw_payload)

    monkeypatch.setattr(openai_client.OpenAICompatClient, "chat_completions", maybe_fail)

    client = TestClient(create_gateway_app(config))
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer tok-keith"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["orchestro_mesh"]["attempts"] == 2

    store = MeshStore(config.store_path)
    with store.connect() as conn:
        statuses = [row["status"] for row in conn.execute("SELECT status FROM jobs ORDER BY id").fetchall()]
    assert "error" in statuses and "completed" in statuses
    assert store.get_node("flaky").current_jobs == 0
    assert store.get_node("local").current_jobs == 0


def test_heartbeat_updates_node_state(gateway):
    client, _ = gateway
    assert client.post("/mesh/heartbeat", json={"node_id": "local"}).status_code == 401
    response = client.post(
        "/mesh/heartbeat",
        headers={"Authorization": "Bearer tok-keith"},
        json={"node_id": "local", "current_jobs": 3, "queue_depth": 7, "status": "degraded"},
    )
    assert response.status_code == 200

    listing = client.get("/mesh/nodes", headers={"Authorization": "Bearer tok-keith"}).json()
    local = next(n for n in listing if n["node_id"] == "local")
    assert local["current_jobs"] == 3
    assert local["queue_depth"] == 7
    assert local["status"] == "degraded"


def test_heartbeat_unknown_node_404(gateway):
    client, _ = gateway
    response = client.post(
        "/mesh/heartbeat",
        headers={"Authorization": "Bearer tok-keith"},
        json={"node_id": "ghost"},
    )
    assert response.status_code == 404


def test_streaming_returns_sse_and_writes_ledger(gateway):
    client, config = gateway
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"Authorization": "Bearer tok-keith"},
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as response:
        assert response.status_code == 200
        body = b"".join(response.iter_bytes())
    assert b"data:" in body
    assert b"[DONE]" in body

    store = MeshStore(config.store_path)
    with store.connect() as conn:
        rows = conn.execute("SELECT status, output_tokens, input_tokens FROM jobs").fetchall()
        ledger = conn.execute("SELECT credit_cost, output_tokens FROM usage_ledger").fetchall()
    assert rows and rows[0]["status"] == "completed"
    assert rows[0]["output_tokens"] >= 1
    assert ledger and ledger[0]["output_tokens"] >= 1
    assert ledger[0]["credit_cost"] >= 0.0


def test_quota_enforced(tmp_path):
    config = MeshConfig(
        local_node_id="local",
        store_path=str(tmp_path / "mesh.db"),
        nodes=[_local_node()],
        api_tokens={"tok-keith": "keith"},
        quota_credits={"keith": 0.0001},
        credits_per_1k_tokens=1000.0,
    )
    client = TestClient(create_gateway_app(config))
    r1 = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer tok-keith"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r1.status_code == 200
    r2 = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer tok-keith"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r2.status_code == 429
    assert r2.json()["detail"]["error"] == "quota exceeded"


def test_context_window_skips_too_small_candidate(tmp_path):
    tiny = _local_node()
    tiny.models[0].context_window = 1  # so small no real prompt fits
    config = MeshConfig(
        local_node_id="local",
        store_path=str(tmp_path / "mesh.db"),
        nodes=[tiny],
        api_tokens={"tok-keith": "keith"},
    )
    client = TestClient(create_gateway_app(config))
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer tok-keith"},
        json={"messages": [{"role": "user", "content": "this prompt is far too long for the tiny window"}]},
    )
    # Policy rejects every candidate → no candidates → 503 with context-window reason.
    assert response.status_code == 503
    body = response.json()
    assert any("context" in r["reason"].lower() for r in body["detail"]["rejections"])


def test_usage_endpoint_reports_totals(gateway):
    client, _ = gateway
    client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer tok-keith"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    response = client.get("/mesh/usage", headers={"Authorization": "Bearer tok-keith"})
    assert response.status_code == 200
    body = response.json()
    assert body["totals"]["keith"] > 0.0
    assert body["credits_per_1k_tokens"] == 1.0


def test_rate_limit_returns_429_with_retry_after(tmp_path):
    config = MeshConfig(
        local_node_id="local",
        store_path=str(tmp_path / "mesh.db"),
        nodes=[_local_node()],
        api_tokens={"tok-keith": "keith"},
        rate_limit_per_minute={"keith": 2},
    )
    client = TestClient(create_gateway_app(config))
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    headers = {"Authorization": "Bearer tok-keith"}
    assert client.post("/v1/chat/completions", headers=headers, json=payload).status_code == 200
    assert client.post("/v1/chat/completions", headers=headers, json=payload).status_code == 200
    blocked = client.post("/v1/chat/completions", headers=headers, json=payload)
    assert blocked.status_code == 429
    assert blocked.headers.get("Retry-After")
    assert blocked.json()["error"] == "rate limit exceeded"


def test_auto_probe_loop_updates_states(tmp_path, monkeypatch):
    from orchestro_mesh import gateway as gateway_module
    from orchestro_mesh.models import ModelState

    calls = {"n": 0}

    async def fake_refresh(node):
        calls["n"] += 1
        for m in node.models:
            m.state = ModelState.ABSENT
        return node

    monkeypatch.setattr(gateway_module, "refresh_node_model_states", fake_refresh)
    config = MeshConfig(
        local_node_id="local",
        store_path=str(tmp_path / "mesh.db"),
        nodes=[_local_node()],
        api_tokens={"tok-keith": "keith"},
        probe_interval_s=0.05,
    )
    app = create_gateway_app(config)
    with TestClient(app):
        import time as _time

        _time.sleep(0.2)
    assert calls["n"] >= 1
    store = MeshStore(config.store_path)
    node = store.get_node("local")
    assert node is not None
    assert all(m.state == ModelState.ABSENT for m in node.models)


def test_probe_endpoint_updates_model_states(gateway, monkeypatch):
    client, _ = gateway
    from orchestro_mesh import gateway as gateway_module
    from orchestro_mesh.models import ModelState

    async def fake_refresh(node):
        for m in node.models:
            m.state = ModelState.ABSENT
        return node

    monkeypatch.setattr(gateway_module, "refresh_node_model_states", fake_refresh)
    response = client.post("/mesh/probe/local", headers={"Authorization": "Bearer tok-keith"})
    assert response.status_code == 200
    body = response.json()
    assert all(m["state"] == "absent" for m in body["models"])
