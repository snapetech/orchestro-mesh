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
)
from orchestro_mesh.policy import evaluate_policy


def make_node(trust_domain="friends", allowed=None):
    return NodeInventory(
        node_id="node-a",
        owner="friend-a",
        trust_domain=trust_domain,
        backends=[BackendEndpoint(id="b1", kind=BackendKind.OPENAI_COMPAT, base_url="http://node-a:8000/v1")],
        models=[ModelCapability(id="m1", backend_id="b1", task_classes=[TaskClass.CHAT, TaskClass.CODING])],
        policy=NodePolicy(
            allowed_users=["keith"],
            allowed_sensitivities=allowed or [Sensitivity.PUBLIC, Sensitivity.FRIENDS_PRIVATE],
            denied_sensitivities=[Sensitivity.SECRET_LOCAL, Sensitivity.TOOL_LOCAL],
        ),
    )


def make_request(sensitivity=Sensitivity.PUBLIC, task_class=TaskClass.CHAT):
    return InferenceRequest(
        requester="keith",
        sensitivity=sensitivity,
        task_class=task_class,
        messages=[ChatMessage(role="user", content="hello")],
    )


def test_public_request_allowed_on_friend_node():
    node = make_node()
    result = evaluate_policy(make_request(), node, node.models[0])
    assert result.allowed is True


def test_secret_local_denied_on_friend_node():
    node = make_node()
    result = evaluate_policy(make_request(Sensitivity.SECRET_LOCAL), node, node.models[0])
    assert result.allowed is False
    assert "denied" in result.reason or "must stay local" in result.reason


def test_tool_use_remote_denied_by_default():
    node = make_node()
    result = evaluate_policy(make_request(Sensitivity.PUBLIC, TaskClass.TOOL_USE), node, node.models[0])
    assert result.allowed is False


def test_local_node_can_allow_secret_local_when_policy_allows_it():
    node = make_node(trust_domain="local", allowed=[Sensitivity.PUBLIC, Sensitivity.SECRET_LOCAL])
    node.policy.denied_sensitivities = []
    result = evaluate_policy(make_request(Sensitivity.SECRET_LOCAL), node, node.models[0])
    assert result.allowed is True
