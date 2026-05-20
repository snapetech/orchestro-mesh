from fastapi.testclient import TestClient

from orchestro_mesh.models import (
    BackendEndpoint,
    BackendKind,
    ModelCapability,
    NodeInventory,
)
from orchestro_mesh.worker import create_worker_app


def _inventory() -> NodeInventory:
    return NodeInventory(
        node_id="w1",
        owner="keith",
        trust_domain="local",
        backends=[BackendEndpoint(id="mock", kind=BackendKind.MOCK, base_url="http://localhost/v1")],
        models=[ModelCapability(id="m1", backend_id="mock")],
    )


def test_inventory_requires_token():
    app = create_worker_app(inventory=_inventory(), worker_token="secret")
    client = TestClient(app)
    assert client.get("/inventory").status_code == 401
    response = client.get("/inventory", headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200
    assert response.json()["node_id"] == "w1"


def test_health_does_not_require_token():
    app = create_worker_app(inventory=_inventory(), worker_token="secret")
    client = TestClient(app)
    assert client.get("/health").status_code == 200


def test_infer_uses_mock_backend():
    app = create_worker_app(inventory=_inventory(), worker_token="secret")
    client = TestClient(app)
    response = client.post(
        "/infer",
        headers={"Authorization": "Bearer secret"},
        json={
            "request": {
                "requester": "keith",
                "messages": [{"role": "user", "content": "hi"}],
            },
            "model_id": "m1",
        },
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "[mock-response]"


def test_open_worker_when_no_token_configured():
    app = create_worker_app(inventory=_inventory(), worker_token=None)
    client = TestClient(app)
    assert client.get("/inventory").status_code == 200
