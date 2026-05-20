from __future__ import annotations

from datetime import datetime

from orchestro_mesh.models import (
    BackendEndpoint,
    InferenceRequest,
    ModelCapability,
    ModelState,
    NodeInventory,
    RouteCandidate,
    RouteDecision,
    RouteRejection,
    RouteResult,
    TaskClass,
    now_utc,
)
from orchestro_mesh.policy import evaluate_policy


class Scheduler:
    # Score nudge applied per (rating - 0.5). A rating of 1.0 → +40, 0.0 → -40.
    RATING_WEIGHT = 80.0

    def __init__(
        self,
        local_node_id: str | None = None,
        node_ttl_seconds: int | None = None,
    ) -> None:
        self.local_node_id = local_node_id
        self.node_ttl_seconds = node_ttl_seconds
        self._rating_averages: dict[tuple[str, str], float] = {}

    def update_rating_averages(self, averages: dict[tuple[str, str], float]) -> None:
        self._rating_averages = dict(averages)

    def route(
        self,
        request: InferenceRequest,
        inventories: list[NodeInventory],
        now: datetime | None = None,
    ) -> RouteResult:
        moment = now or now_utc()
        candidates: list[RouteCandidate] = []
        rejections: list[RouteRejection] = []

        for node in inventories:
            if self._is_stale(node, moment):
                rejections.append(RouteRejection(node_id=node.node_id, reason="node heartbeat is stale"))
                continue
            for model in node.models:
                backend = node.backend_by_id(model.backend_id)
                if backend is None:
                    rejections.append(RouteRejection(node_id=node.node_id, model_id=model.id, reason="model backend is missing"))
                    continue

                policy = evaluate_policy(request, node, model)
                if not policy.allowed:
                    rejections.append(RouteRejection(node_id=node.node_id, model_id=model.id, reason=policy.reason))
                    continue

                score, reasons = self.score(request, node, model, backend)
                candidates.append(RouteCandidate(node=node, model=model, backend=backend, score=score, reasons=reasons))

        if not candidates:
            return RouteResult(decision=RouteDecision.DENY, request_id=request.id, rejections=rejections)

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return RouteResult(
            decision=RouteDecision.ALLOW,
            request_id=request.id,
            selected=candidates[0],
            candidates=candidates,
            rejections=rejections,
        )

    def _is_stale(self, node: NodeInventory, now: datetime) -> bool:
        if self.node_ttl_seconds is None or self.node_ttl_seconds <= 0:
            return False
        if self.local_node_id and node.node_id == self.local_node_id:
            return False
        age = (now - node.last_seen).total_seconds()
        return age > self.node_ttl_seconds

    def score(
        self,
        request: InferenceRequest,
        node: NodeInventory,
        model: ModelCapability,
        backend: BackendEndpoint,
    ) -> tuple[float, list[str]]:
        score = 0.0
        reasons: list[str] = []

        if request.model and request.model == model.id:
            score += 1000.0
            reasons.append("exact model requested")

        if request.task_class in model.task_classes:
            score += 120.0
            reasons.append(f"task class {request.task_class.value} advertised")

        tag_hits = len(set(request.required_tags).intersection(set(model.tags)))
        if tag_hits:
            score += 20.0 * tag_hits
            reasons.append(f"{tag_hits} required tag match(es)")

        if model.state == ModelState.ACTIVE:
            score += 90.0
            reasons.append("model active")
        elif model.state == ModelState.WARM:
            score += 75.0
            reasons.append("model warm")
        elif model.state == ModelState.LOADING:
            score -= 25.0
            reasons.append("model loading")
        elif model.state == ModelState.COLD:
            score -= 50.0
            reasons.append("model cold")

        if self.local_node_id and node.node_id == self.local_node_id:
            score += 45.0
            reasons.append("local node bonus")

        queue_penalty = node.queue_depth * 8.0 + node.current_jobs * 12.0
        if queue_penalty:
            score -= queue_penalty
            reasons.append(f"queue/current job penalty {queue_penalty:.1f}")

        if request.task_class == TaskClass.TOOL_USE:
            tps = model.benchmark.best_tool_output_tps()
            score += min(tps, 250.0) * 1.5
            reasons.append(f"tool-output benchmark {tps:.2f} tok/s")
        else:
            tps = model.benchmark.best_decode_tps()
            score += min(tps, 250.0)
            reasons.append(f"decode benchmark {tps:.2f} tok/s")

        if backend.supports_streaming and request.stream:
            score += 15.0
            reasons.append("streaming supported")

        if backend.supports_tools and request.task_class == TaskClass.TOOL_USE:
            score += 30.0
            reasons.append("backend advertises tool support")

        score += model.benchmark.confidence * 20.0

        rating = self._rating_averages.get((node.node_id, model.id))
        if rating is not None:
            nudge = (rating - 0.5) * self.RATING_WEIGHT
            score += nudge
            reasons.append(f"feedback nudge {nudge:+.1f} (avg {rating:.2f})")

        return score, reasons
