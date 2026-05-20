from fastapi.testclient import TestClient

from orchestro_mesh.models import (
    BackendEndpoint,
    BackendKind,
    ModelCapability,
    NodeInventory,
    NodePolicy,
    Sensitivity,
)
from orchestro_mesh.worker import create_worker_app


def _inventory() -> NodeInventory:
    return NodeInventory(
        node_id="w1",
        owner="keith",
        trust_domain="local",
        backends=[BackendEndpoint(id="mock", kind=BackendKind.MOCK, base_url="http://x/v1")],
        models=[ModelCapability(id="m1", backend_id="mock")],
        policy=NodePolicy(
            allowed_users=[],
            allowed_sensitivities=[Sensitivity.PUBLIC, Sensitivity.SECRET_LOCAL, Sensitivity.TOOL_LOCAL],
            denied_sensitivities=[],
            max_concurrent_jobs=4,
            allow_remote_tool_authority=True,
        ),
    )


def _client_with_lifespan(token: str = "secret"):
    app = create_worker_app(inventory=_inventory(), worker_token=token)
    return TestClient(app)


def test_drain_blocks_new_infer():
    with _client_with_lifespan() as client:
        assert client.post(
            "/drain", headers={"Authorization": "Bearer secret"}
        ).status_code == 200

        response = client.post(
            "/infer",
            headers={"Authorization": "Bearer secret"},
            json={
                "request": {"requester": "keith", "messages": [{"role": "user", "content": "hi"}]},
                "model_id": "m1",
            },
        )
        assert response.status_code == 503
        assert "draining" in response.json()["detail"]


def test_resume_clears_drain():
    with _client_with_lifespan() as client:
        headers = {"Authorization": "Bearer secret"}
        client.post("/drain", headers=headers)
        assert client.post("/resume", headers=headers).status_code == 200
        response = client.post(
            "/infer",
            headers=headers,
            json={
                "request": {"requester": "keith", "messages": [{"role": "user", "content": "hi"}]},
                "model_id": "m1",
            },
        )
        assert response.status_code == 200


def test_worker_policy_blocks_remote_tool_authority():
    # Override inventory so allow_remote_tool_authority is False; sensitivity allowed.
    inventory = _inventory()
    inventory.trust_domain = "friends"
    inventory.policy.allow_remote_tool_authority = False
    app = create_worker_app(inventory=inventory, worker_token="secret")
    with TestClient(app) as client:
        response = client.post(
            "/infer",
            headers={"Authorization": "Bearer secret"},
            json={
                "request": {
                    "requester": "keith",
                    "messages": [{"role": "user", "content": "hi"}],
                    "tool_authority_required": True,
                },
                "model_id": "m1",
            },
        )
        assert response.status_code == 403


def test_health_reflects_drain_and_in_flight():
    with _client_with_lifespan() as client:
        response = client.get("/health").json()
        assert response["draining"] is False
        assert response["in_flight"] == 0
        client.post("/drain", headers={"Authorization": "Bearer secret"})
        response = client.get("/health").json()
        assert response["draining"] is True
        assert response["ok"] is False
