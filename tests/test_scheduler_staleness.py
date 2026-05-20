from datetime import timedelta

from orchestro_mesh.models import (
    BackendEndpoint,
    BackendKind,
    ChatMessage,
    InferenceRequest,
    ModelCapability,
    NodeInventory,
    NodePolicy,
    Sensitivity,
    TaskClass,
    now_utc,
)
from orchestro_mesh.scheduler import Scheduler


def _node(node_id: str, last_seen_age_s: float) -> NodeInventory:
    return NodeInventory(
        node_id=node_id,
        owner="keith",
        trust_domain="friends",
        backends=[BackendEndpoint(id="b", kind=BackendKind.MOCK, base_url="http://x/v1")],
        models=[
            ModelCapability(
                id=f"{node_id}-m",
                backend_id="b",
                task_classes=[TaskClass.CHAT],
            )
        ],
        policy=NodePolicy(
            allowed_sensitivities=[Sensitivity.PUBLIC],
            denied_sensitivities=[],
            max_concurrent_jobs=4,
        ),
        last_seen=now_utc() - timedelta(seconds=last_seen_age_s),
    )


def _request() -> InferenceRequest:
    return InferenceRequest(
        requester="keith",
        messages=[ChatMessage(role="user", content="hi")],
    )


def test_stale_node_is_filtered_when_ttl_set():
    scheduler = Scheduler(local_node_id=None, node_ttl_seconds=60)
    fresh = _node("fresh", last_seen_age_s=10)
    stale = _node("stale", last_seen_age_s=600)
    result = scheduler.route(_request(), [fresh, stale])
    assert result.selected is not None
    assert result.selected.node.node_id == "fresh"
    assert any(r.node_id == "stale" and "stale" in r.reason for r in result.rejections)


def test_stale_local_node_is_exempt():
    scheduler = Scheduler(local_node_id="me", node_ttl_seconds=60)
    me = _node("me", last_seen_age_s=99999)
    result = scheduler.route(_request(), [me])
    assert result.selected is not None
    assert result.selected.node.node_id == "me"


def test_ttl_disabled_does_not_filter():
    scheduler = Scheduler(local_node_id=None, node_ttl_seconds=None)
    stale = _node("stale", last_seen_age_s=99999)
    result = scheduler.route(_request(), [stale])
    assert result.selected is not None


def test_candidates_list_is_populated_and_ranked():
    scheduler = Scheduler(local_node_id=None, node_ttl_seconds=None)
    a = _node("a", last_seen_age_s=10)
    b = _node("b", last_seen_age_s=10)
    a.models[0].benchmark.expected_decode_tps = 100
    b.models[0].benchmark.expected_decode_tps = 10
    result = scheduler.route(_request(), [b, a])
    assert [c.node.node_id for c in result.candidates] == ["a", "b"]
