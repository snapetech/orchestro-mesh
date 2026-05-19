from __future__ import annotations

from dataclasses import dataclass

from orchestro_mesh.models import InferenceRequest, ModelCapability, NodeInventory, NodeStatus, Sensitivity, TaskClass


@dataclass(frozen=True)
class PolicyResult:
    allowed: bool
    reason: str = "allowed"


def evaluate_policy(request: InferenceRequest, node: NodeInventory, model: ModelCapability) -> PolicyResult:
    if node.status not in {NodeStatus.ONLINE, NodeStatus.DEGRADED}:
        return PolicyResult(False, f"node status is {node.status.value}")

    if node.current_jobs >= node.policy.max_concurrent_jobs:
        return PolicyResult(False, "node concurrency limit reached")

    if node.policy.allowed_users and request.requester not in node.policy.allowed_users:
        return PolicyResult(False, "requester is not allowed on node")

    if request.sensitivity in node.policy.denied_sensitivities:
        return PolicyResult(False, f"sensitivity {request.sensitivity.value} is denied by node")

    if request.sensitivity not in node.policy.allowed_sensitivities:
        return PolicyResult(False, f"sensitivity {request.sensitivity.value} is not allowed by node")

    if request.sensitivity in {Sensitivity.SECRET_LOCAL, Sensitivity.TOOL_LOCAL} and node.trust_domain != "local":
        return PolicyResult(False, f"{request.sensitivity.value} must stay local")

    if request.tool_authority_required and not node.policy.allow_remote_tool_authority:
        return PolicyResult(False, "remote tool authority is disabled")

    if request.task_class == TaskClass.TOOL_USE and node.trust_domain != "local":
        return PolicyResult(False, "tool-use jobs are local-only by default")

    if request.max_context_tokens and model.context_window and request.max_context_tokens > model.context_window:
        return PolicyResult(False, "requested context exceeds model context window")

    if node.policy.max_context_tokens and request.max_context_tokens and request.max_context_tokens > node.policy.max_context_tokens:
        return PolicyResult(False, "requested context exceeds node policy")

    if request.max_output_tokens and model.max_output_tokens and request.max_output_tokens > model.max_output_tokens:
        return PolicyResult(False, "requested output exceeds model limit")

    if node.policy.max_output_tokens and request.max_output_tokens and request.max_output_tokens > node.policy.max_output_tokens:
        return PolicyResult(False, "requested output exceeds node output policy")

    missing_tags = [tag for tag in request.required_tags if tag not in model.tags and tag not in node.policy.tags]
    if missing_tags:
        return PolicyResult(False, f"missing required tags: {', '.join(missing_tags)}")

    blocked_tags = [tag for tag in request.forbidden_tags if tag in model.tags or tag in node.policy.tags]
    if blocked_tags:
        return PolicyResult(False, f"blocked tags present: {', '.join(blocked_tags)}")

    if request.task_class not in model.task_classes and request.task_class != TaskClass.CHAT:
        return PolicyResult(False, f"model does not advertise task class {request.task_class.value}")

    return PolicyResult(True)
