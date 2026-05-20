from __future__ import annotations

import base64

from fastapi.testclient import TestClient

from orchestro_mesh.config import MeshConfig
from orchestro_mesh.gateway import create_gateway_app
from orchestro_mesh.models import (
    BackendEndpoint,
    BackendKind,
    FeedbackRating,
    ModelCapability,
    ModelState,
    NodeInventory,
    NodePolicy,
    Sensitivity,
    TaskClass,
)
from orchestro_mesh.store import MeshStore


def _node(node_id: str) -> NodeInventory:
    return NodeInventory(
        node_id=node_id,
        owner="keith",
        trust_domain="local",
        backends=[BackendEndpoint(id=f"{node_id}-mock", kind=BackendKind.MOCK, base_url="http://x/v1")],
        models=[
            ModelCapability(
                id=f"{node_id}-m",
                backend_id=f"{node_id}-mock",
                state=ModelState.WARM,
                context_window=32768,
                task_classes=[TaskClass.CHAT, TaskClass.CODING],
            )
        ],
        policy=NodePolicy(
            allowed_users=[],
            allowed_sensitivities=[Sensitivity.PUBLIC],
            denied_sensitivities=[],
            max_concurrent_jobs=2,
        ),
    )


def _gateway(tmp_path):
    config = MeshConfig(
        local_node_id="local",
        store_path=str(tmp_path / "mesh.db"),
        nodes=[_node("local"), _node("friend")],
        api_tokens={"tok-keith": "keith"},
        quota_credits={"keith": 10.0},
    )
    return TestClient(create_gateway_app(config)), config


def test_dashboard_requires_auth(tmp_path):
    client, _ = _gateway(tmp_path)
    response = client.get("/dashboard")
    assert response.status_code == 401
    assert "Basic" in response.headers.get("WWW-Authenticate", "")
    assert "Bearer" in response.headers.get("WWW-Authenticate", "")


def test_dashboard_renders_with_bearer(tmp_path):
    client, _ = _gateway(tmp_path)
    response = client.get("/dashboard", headers={"Authorization": "Bearer tok-keith"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert "Orchestro Mesh" in body
    assert "local" in body
    assert "friend" in body
    assert "Recent jobs" in body
    assert "Usage" in body
    assert "Recent feedback" in body


def test_dashboard_renders_with_basic_auth(tmp_path):
    client, _ = _gateway(tmp_path)
    creds = base64.b64encode(b"keith:tok-keith").decode()
    response = client.get("/dashboard", headers={"Authorization": f"Basic {creds}"})
    assert response.status_code == 200
    assert "Orchestro Mesh" in response.text


def test_dashboard_basic_auth_rejects_bad_password(tmp_path):
    client, _ = _gateway(tmp_path)
    creds = base64.b64encode(b"keith:wrong").decode()
    response = client.get("/dashboard", headers={"Authorization": f"Basic {creds}"})
    assert response.status_code == 401


def test_dashboard_shows_jobs_and_feedback_after_activity(tmp_path):
    client, config = _gateway(tmp_path)
    # Drive one completion to generate a job + ledger entry
    client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer tok-keith"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    # Seed feedback directly
    store = MeshStore(config.store_path)
    store.add_feedback(FeedbackRating(job_id="seed", node_id="local", model_id="local-m", rating=0.85, verifier="lint"))

    response = client.get("/dashboard", headers={"Authorization": "Bearer tok-keith"})
    body = response.text
    assert "completed" in body  # job status rendered
    assert "0.85" in body  # rating value
    assert "lint" in body  # verifier name
    assert "keith" in body  # requester


def test_basic_auth_username_becomes_requester_in_mesh_token_mode(tmp_path):
    config = MeshConfig(
        local_node_id="local",
        store_path=str(tmp_path / "mesh.db"),
        nodes=[_node("local")],
        mesh_token="shared",
    )
    client = TestClient(create_gateway_app(config))
    creds = base64.b64encode(b"alice:shared").decode()
    # alice should show up in the dashboard usage section if she submits a request
    client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Basic {creds}"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    response = client.get("/dashboard", headers={"Authorization": f"Basic {creds}"})
    assert response.status_code == 200
    assert "alice" in response.text


def test_dashboard_empty_state(tmp_path):
    config = MeshConfig(
        local_node_id=None,
        store_path=str(tmp_path / "mesh.db"),
        nodes=[],
        api_tokens={"tok-keith": "keith"},
    )
    client = TestClient(create_gateway_app(config))
    response = client.get("/dashboard", headers={"Authorization": "Bearer tok-keith"})
    assert response.status_code == 200
    body = response.text
    assert "no nodes registered" in body
    assert "no jobs yet" in body
    assert "no feedback yet" in body
