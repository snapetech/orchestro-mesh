from __future__ import annotations

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


def _make_client(tmp_path, nodes):
    config = MeshConfig(
        local_node_id=None,
        store_path=str(tmp_path / "mesh.db"),
        nodes=nodes,
        api_tokens={"tok-keith": "keith", "tok-other": "other"},
    )
    return TestClient(create_gateway_app(config)), config


def test_request_id_surfaced_in_response(tmp_path):
    client, _ = _make_client(tmp_path, [_node("local")])
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer tok-keith"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["orchestro_mesh"]["request_id"]
    assert body["orchestro_mesh"]["node_id"] == "local"
    assert body["orchestro_mesh"]["model_id"] == "local-m"


def test_feedback_round_trip(tmp_path):
    client, config = _make_client(tmp_path, [_node("local")])
    completion = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer tok-keith"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    ).json()
    job_id = completion["orchestro_mesh"]["request_id"]

    response = client.post(
        f"/v1/feedback/{job_id}",
        headers={"Authorization": "Bearer tok-keith"},
        json={"rating": 0.9, "verifier": "python-syntax", "notes": "looks fine"},
    )
    assert response.status_code == 200
    fetched = client.get(
        f"/v1/feedback/{job_id}",
        headers={"Authorization": "Bearer tok-keith"},
    ).json()
    assert len(fetched["ratings"]) == 1
    assert fetched["ratings"][0]["rating"] == 0.9
    assert fetched["ratings"][0]["verifier"] == "python-syntax"

    store = MeshStore(config.store_path)
    averages = store.recent_rating_averages()
    assert averages[("local", "local-m")] == 0.9


def test_feedback_rejects_other_requesters_job(tmp_path):
    client, _ = _make_client(tmp_path, [_node("local")])
    completion = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer tok-keith"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    ).json()
    job_id = completion["orchestro_mesh"]["request_id"]
    response = client.post(
        f"/v1/feedback/{job_id}",
        headers={"Authorization": "Bearer tok-other"},
        json={"rating": 0.0},
    )
    assert response.status_code == 403


def test_feedback_rating_out_of_range(tmp_path):
    client, _ = _make_client(tmp_path, [_node("local")])
    completion = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer tok-keith"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    ).json()
    job_id = completion["orchestro_mesh"]["request_id"]
    response = client.post(
        f"/v1/feedback/{job_id}",
        headers={"Authorization": "Bearer tok-keith"},
        json={"rating": 2.5},
    )
    assert response.status_code == 400


def test_scheduler_nudges_score_toward_higher_rated_pair(tmp_path):
    nodes = [_node("good"), _node("bad")]
    # Same advertised TPS — make them otherwise identical so only feedback drives choice.
    client, config = _make_client(tmp_path, nodes)
    # Seed feedback by posting to two completions, one for each node.
    # We can't pin the choice deterministically without forcing it, so we'll seed via the store.
    store = MeshStore(config.store_path)
    from orchestro_mesh.models import FeedbackRating

    for _ in range(10):
        store.add_feedback(FeedbackRating(job_id="x", node_id="good", model_id="good-m", rating=1.0))
        store.add_feedback(FeedbackRating(job_id="x", node_id="bad", model_id="bad-m", rating=0.0))

    # New gateway picks up averages at startup.
    client2 = TestClient(create_gateway_app(config))
    decision = client2.post(
        "/mesh/route",
        headers={"Authorization": "Bearer tok-keith"},
        json={"messages": [{"role": "user", "content": "hi"}], "requester": "keith"},
    ).json()
    assert decision["selected"]["node"]["node_id"] == "good"
    reasons = " ".join(decision["selected"]["reasons"])
    assert "feedback nudge" in reasons
