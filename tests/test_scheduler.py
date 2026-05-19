from orchestro_mesh.models import (
    BackendEndpoint,
    BenchmarkProfile,
    ChatMessage,
    InferenceRequest,
    ModelCapability,
    ModelState,
    NodeInventory,
    NodePolicy,
    Sensitivity,
    TaskClass,
)
from orchestro_mesh.scheduler import Scheduler


def node(node_id, trust_domain, tps, state=ModelState.WARM, queue_depth=0):
    return NodeInventory(
        node_id=node_id,
        owner=node_id,
        trust_domain=trust_domain,
        queue_depth=queue_depth,
        backends=[BackendEndpoint(id="backend", base_url=f"http://{node_id}:8000/v1")],
        models=[
            ModelCapability(
                id=f"{node_id}-model",
                backend_id="backend",
                state=state,
                tags=["coding"],
                task_classes=[TaskClass.CHAT, TaskClass.CODING, TaskClass.TOOL_USE],
                benchmark=BenchmarkProfile(expected_decode_tps=tps, expected_tool_output_tps=tps / 2, confidence=0.5),
            )
        ],
        policy=NodePolicy(
            allowed_users=["keith"],
            allowed_sensitivities=[Sensitivity.PUBLIC, Sensitivity.FRIENDS_PRIVATE],
            denied_sensitivities=[Sensitivity.SECRET_LOCAL, Sensitivity.TOOL_LOCAL],
            max_concurrent_jobs=2,
        ),
    )


def request(task=TaskClass.CODING):
    return InferenceRequest(
        requester="keith",
        task_class=task,
        messages=[ChatMessage(role="user", content="fix this")],
    )


def test_scheduler_prefers_faster_warm_candidate():
    scheduler = Scheduler(local_node_id="local")
    result = scheduler.route(request(), [node("slow", "friends", 10), node("fast", "friends", 80)])
    assert result.selected is not None
    assert result.selected.node.node_id == "fast"


def test_scheduler_respects_secret_local():
    scheduler = Scheduler(local_node_id="local")
    req = InferenceRequest(
        requester="keith",
        sensitivity=Sensitivity.SECRET_LOCAL,
        messages=[ChatMessage(role="user", content="secret")],
    )
    result = scheduler.route(req, [node("remote", "friends", 100)])
    assert result.selected is None
    assert result.rejections


def test_tool_output_tps_affects_tool_use_score():
    scheduler = Scheduler(local_node_id=None)
    fast_tool = node("toolfast", "local", 40)
    slow_tool = node("toolslow", "local", 20)
    fast_tool.models[0].benchmark.expected_tool_output_tps = 80
    slow_tool.models[0].benchmark.expected_tool_output_tps = 5
    result = scheduler.route(request(TaskClass.TOOL_USE), [slow_tool, fast_tool])
    assert result.selected is not None
    assert result.selected.node.node_id == "toolfast"
