"""End-to-end test exercising the orchestro ↔ orchestro-mesh contract.

Boots a real uvicorn process serving the mesh gateway, then has a real
MeshBackend (from the sibling ``orchestro`` package) talk to it. Verifies:

- The mesh request_id is captured into BackendResponse.metadata.
- mesh_feedback.maybe_report POSTs feedback and the gateway stores it.
- The scheduler's rating nudge takes effect on a subsequent route.

Skipped automatically if the ``orchestro`` package is not installed
alongside (e.g. local-only mesh dev). CI installs both repos so the test
runs there.
"""

from __future__ import annotations

import socket
import threading
import time
import urllib.request

import pytest

orchestro = pytest.importorskip("orchestro")
mesh_backend_mod = pytest.importorskip("orchestro.backends.mesh")
mesh_feedback_mod = pytest.importorskip("orchestro.mesh_feedback")

import uvicorn  # noqa: E402  (after importorskip)

from orchestro_mesh.config import MeshConfig  # noqa: E402
from orchestro_mesh.gateway import create_gateway_app  # noqa: E402
from orchestro_mesh.models import (  # noqa: E402
    BackendEndpoint,
    BackendKind,
    ModelCapability,
    ModelState,
    NodeInventory,
    NodePolicy,
    Sensitivity,
    TaskClass,
)
from orchestro_mesh.store import MeshStore  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_ready(url: str, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError(f"gateway never became ready at {url}")


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
                task_classes=[TaskClass.CHAT, TaskClass.CODING],
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


@pytest.fixture
def live_gateway(tmp_path):
    config = MeshConfig(
        local_node_id="local",
        store_path=str(tmp_path / "mesh.db"),
        nodes=[_node("local"), _node("friend")],
        api_tokens={"tok-keith": "keith"},
    )
    app = create_gateway_app(config)

    port = _free_port()
    uv_config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(uv_config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        _wait_ready(f"http://127.0.0.1:{port}/health")
        yield {
            "url": f"http://127.0.0.1:{port}",
            "config": config,
            "token": "tok-keith",
        }
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_mesh_backend_run_round_trip(live_gateway):
    from orchestro.models import RunRequest

    backend = mesh_backend_mod.MeshBackend(
        gateway_url=live_gateway["url"],
        token=live_gateway["token"],
        requester="keith",
        model="local-m",
    )
    response = backend.run(
        RunRequest(goal="say hi", backend_name="mesh", metadata={"backend_model": "local-m"})
    )
    assert response.output_text == "[mock-response]"
    # Mesh metadata propagated end-to-end via real HTTP.
    assert response.metadata["mesh_request_id"]
    assert response.metadata["mesh_node_id"] in {"local", "friend"}
    assert response.metadata["mesh_model_id"] in {"local-m", "friend-m"}
    assert response.metadata["mesh_gateway_url"] == f"{live_gateway['url']}/v1"


def test_feedback_round_trip_via_real_gateway(live_gateway):
    from orchestro.models import RunRequest
    from orchestro.verifiers import VerificationResult

    backend = mesh_backend_mod.MeshBackend(
        gateway_url=live_gateway["url"],
        token=live_gateway["token"],
        requester="keith",
        model="local-m",
    )
    response = backend.run(
        RunRequest(goal="say hi", backend_name="mesh", metadata={"backend_model": "local-m"})
    )
    request_id = response.metadata["mesh_request_id"]

    results = [
        VerificationResult(passed=True, verifier="python-syntax", errors=[], warnings=[]),
        VerificationResult(passed=False, verifier="json-structure", errors=["bad"], warnings=[]),
    ]
    sent = mesh_feedback_mod.maybe_report(response, results, timeout_s=5.0)
    assert sent is True

    store = MeshStore(live_gateway["config"].store_path)
    ratings = store.list_feedback(request_id)
    assert len(ratings) == 1
    assert ratings[0].rating == 0.5
    assert "python-syntax" in ratings[0].verifier
    assert "json-structure" in ratings[0].verifier


def test_scheduler_nudge_picks_higher_rated_pair_after_feedback(tmp_path):
    """Seed feedback in the store, then start a fresh gateway and verify routing.

    Uses /mesh/route (which doesn't dispatch to a backend) so we observe the
    pure routing decision deterministically.
    """
    from orchestro.models import RunRequest  # noqa: F401 — proves orchestro is importable

    config = MeshConfig(
        local_node_id=None,
        store_path=str(tmp_path / "mesh.db"),
        nodes=[_node("good"), _node("bad")],
        api_tokens={"tok-keith": "keith"},
    )

    # Seed feedback directly in the store (bypass HTTP).
    store = MeshStore(config.store_path)
    from orchestro_mesh.models import FeedbackRating

    for _ in range(10):
        store.add_feedback(FeedbackRating(job_id="seed", node_id="good", model_id="good-m", rating=1.0))
        store.add_feedback(FeedbackRating(job_id="seed", node_id="bad", model_id="bad-m", rating=0.0))

    app = create_gateway_app(config)
    port = _free_port()
    uv_config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(uv_config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        _wait_ready(f"http://127.0.0.1:{port}/health")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/mesh/route",
            data=b'{"requester":"keith","messages":[{"role":"user","content":"hi"}]}',
            headers={
                "Authorization": "Bearer tok-keith",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            import json as _json

            body = _json.loads(resp.read().decode("utf-8"))
    finally:
        server.should_exit = True
        thread.join(timeout=5)

    assert body["selected"]["node"]["node_id"] == "good"
    assert any("feedback nudge" in r for r in body["selected"]["reasons"])
