from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from orchestro_mesh.auth import make_requester_dep
from orchestro_mesh.config import MeshConfig, apply_env_overrides, load_config
from orchestro_mesh.dashboard import render_dashboard
from orchestro_mesh.models import (
    ChatMessage,
    FeedbackRating,
    Heartbeat,
    InferenceRequest,
    JobRecord,
    RouteCandidate,
    Sensitivity,
    TaskClass,
    UsageLedgerEntry,
    now_utc,
)
from orchestro_mesh.openai_client import OpenAICompatClient
from orchestro_mesh.probe import refresh_node_model_states
from orchestro_mesh.rate_limit import SlidingWindowLimiter
from orchestro_mesh.redaction import redact_text
from orchestro_mesh.scheduler import Scheduler
from orchestro_mesh.store import MeshStore
from orchestro_mesh.tokens import count_message_tokens, count_text_tokens

logger = logging.getLogger(__name__)


def load_gateway_config() -> MeshConfig:
    config_path = os.environ.get("ORCHESTRO_MESH_CONFIG")
    config = load_config(config_path) if config_path else MeshConfig()
    return apply_env_overrides(config)


def _scan_messages_for_secrets(messages: list[ChatMessage]) -> list[str]:
    findings: list[str] = []
    for message in messages:
        content = message.content
        if isinstance(content, str):
            findings.extend(redact_text(content).findings)
        elif isinstance(content, list):
            for part in content:
                text = part.get("text") if isinstance(part, dict) else None
                if isinstance(text, str):
                    findings.extend(redact_text(text).findings)
    return findings


def _has_tools(payload: dict[str, Any]) -> bool:
    tools = payload.get("tools")
    if isinstance(tools, list) and tools:
        return True
    choice = payload.get("tool_choice")
    if isinstance(choice, str) and choice not in {"", "none"}:
        return True
    if isinstance(choice, dict):
        return True
    return False


def _is_transient_backend_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError | httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status >= 500 or status == 429
    return False


def _credit_cost(input_tokens: int, output_tokens: int, credits_per_1k: float) -> float:
    return ((input_tokens + output_tokens) / 1000.0) * credits_per_1k


class _StreamCounter:
    """Accumulates output text by parsing OpenAI-style SSE chunks."""

    def __init__(self) -> None:
        self.text = ""
        self._buffer = b""

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk
        while b"\n\n" in self._buffer:
            event, self._buffer = self._buffer.split(b"\n\n", 1)
            for line in event.split(b"\n"):
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if not data or data == b"[DONE]":
                    continue
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or choice.get("message") or {}
                    content = delta.get("content")
                    if isinstance(content, str):
                        self.text += content


async def _probe_loop(store: MeshStore, interval_s: float, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            for node in store.list_nodes():
                refreshed = await refresh_node_model_states(node)
                store.upsert_node(refreshed)
        except Exception as exc:  # noqa: BLE001
            logger.warning("probe.loop_error error=%s", exc)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            continue


def create_gateway_app(config: MeshConfig | None = None) -> FastAPI:
    config = config or load_gateway_config()
    store = MeshStore(config.store_path)
    for node in config.nodes:
        store.upsert_node(node)
    scheduler = Scheduler(
        local_node_id=config.local_node_id,
        node_ttl_seconds=config.node_ttl_seconds,
    )
    scheduler.update_rating_averages(store.recent_rating_averages())
    requester_dep = make_requester_dep(
        mesh_token=config.mesh_token,
        api_tokens=config.api_tokens,
    )
    limiter = SlidingWindowLimiter(
        per_minute=config.rate_limit_per_minute,
        default_per_minute=config.rate_limit_default_per_minute,
    )

    probe_state: dict[str, Any] = {"stop": None, "task": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if config.probe_interval_s and config.probe_interval_s > 0:
            stop = asyncio.Event()
            probe_state["stop"] = stop
            probe_state["task"] = asyncio.create_task(_probe_loop(store, config.probe_interval_s, stop))
            logger.info("gateway.probe_loop.started interval_s=%.1f", config.probe_interval_s)
        try:
            yield
        finally:
            stop = probe_state.get("stop")
            task = probe_state.get("task")
            if stop is not None:
                stop.set()
            if task is not None:
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except asyncio.TimeoutError:
                    task.cancel()

    app = FastAPI(title="Orchestro Mesh Gateway", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "nodes": len(store.list_nodes())}

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(requester: str = Depends(requester_dep)) -> HTMLResponse:
        html_body = render_dashboard(store=store, config=config)
        return HTMLResponse(content=html_body)

    @app.get("/mesh/nodes")
    async def list_nodes(requester: str = Depends(requester_dep)) -> list[dict[str, Any]]:
        return [node.model_dump(mode="json") for node in store.list_nodes()]

    @app.post("/mesh/nodes")
    async def register_node(
        node: dict[str, Any],
        requester: str = Depends(requester_dep),
    ) -> dict[str, Any]:
        from orchestro_mesh.models import NodeInventory

        inventory = NodeInventory.model_validate(node)
        inventory.last_seen = now_utc()
        store.upsert_node(inventory)
        logger.info("mesh.node.registered node_id=%s requester=%s", inventory.node_id, requester)
        return {"ok": True, "node_id": inventory.node_id}

    @app.post("/mesh/heartbeat")
    async def heartbeat(
        beat: Heartbeat,
        requester: str = Depends(requester_dep),
    ) -> dict[str, Any]:
        node = store.get_node(beat.node_id)
        if node is None:
            raise HTTPException(status_code=404, detail=f"unknown node {beat.node_id}")
        node.last_seen = now_utc()
        if beat.status is not None:
            node.status = beat.status
        if beat.current_jobs is not None:
            node.current_jobs = max(0, beat.current_jobs)
        if beat.queue_depth is not None:
            node.queue_depth = max(0, beat.queue_depth)
        store.upsert_node(node)
        return {"ok": True, "node_id": node.node_id, "last_seen": node.last_seen.isoformat()}

    @app.post("/mesh/probe/{node_id}")
    async def probe_node(
        node_id: str,
        requester: str = Depends(requester_dep),
    ) -> dict[str, Any]:
        node = store.get_node(node_id)
        if node is None:
            raise HTTPException(status_code=404, detail=f"unknown node {node_id}")
        await refresh_node_model_states(node)
        store.upsert_node(node)
        return {
            "ok": True,
            "node_id": node.node_id,
            "models": [{"id": m.id, "state": m.state.value} for m in node.models],
        }

    @app.get("/mesh/usage")
    async def usage(requester: str = Depends(requester_dep)) -> dict[str, Any]:
        return {
            "totals": store.ledger_totals(),
            "quotas": config.quota_credits,
            "credits_per_1k_tokens": config.credits_per_1k_tokens,
        }

    @app.post("/v1/feedback/{job_id}")
    async def submit_feedback(
        job_id: str,
        payload: dict[str, Any],
        requester: str = Depends(requester_dep),
    ) -> dict[str, Any]:
        rating_value = payload.get("rating")
        if rating_value is None:
            raise HTTPException(status_code=400, detail="rating is required")
        try:
            rating_value = float(rating_value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="rating must be a number in [0, 1]") from None
        if not (0.0 <= rating_value <= 1.0):
            raise HTTPException(status_code=400, detail="rating must be in [0, 1]")

        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
        if job.requester != requester:
            raise HTTPException(status_code=403, detail="job belongs to a different requester")

        rating = FeedbackRating(
            job_id=job_id,
            node_id=job.node_id,
            model_id=job.model_id,
            requester=requester,
            verifier=str(payload.get("verifier") or "human"),
            rating=rating_value,
            notes=payload.get("notes"),
        )
        store.add_feedback(rating)
        scheduler.update_rating_averages(store.recent_rating_averages())
        logger.info(
            "feedback.received job_id=%s requester=%s verifier=%s rating=%.3f node=%s model=%s",
            job_id,
            requester,
            rating.verifier,
            rating_value,
            job.node_id,
            job.model_id,
        )
        return {"ok": True, "id": rating.id}

    @app.get("/v1/feedback/{job_id}")
    async def list_feedback(
        job_id: str,
        requester: str = Depends(requester_dep),
    ) -> dict[str, Any]:
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
        if job.requester != requester:
            raise HTTPException(status_code=403, detail="job belongs to a different requester")
        return {
            "job_id": job_id,
            "ratings": [r.model_dump(mode="json") for r in store.list_feedback(job_id)],
        }

    @app.post("/mesh/route")
    async def route(
        request: InferenceRequest,
        requester: str = Depends(requester_dep),
    ) -> dict[str, Any]:
        result = scheduler.route(request, store.list_nodes())
        return result.model_dump(mode="json")

    @app.post("/v1/chat/completions")
    async def chat_completions(
        payload: dict[str, Any],
        requester: str = Depends(requester_dep),
    ) -> Any:
        allowed, limit, retry_after = limiter.check(requester)
        if not allowed:
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(int((retry_after or 1.0) + 0.5))},
                content={
                    "error": "rate limit exceeded",
                    "requester": requester,
                    "limit_per_minute": limit,
                    "retry_after_s": retry_after,
                },
            )

        quota = config.quota_credits.get(requester)
        if quota is not None:
            used = store.ledger_totals().get(requester, 0.0)
            if used >= quota:
                raise HTTPException(
                    status_code=429,
                    detail={"error": "quota exceeded", "requester": requester, "used": used, "quota": quota},
                )

        messages = [ChatMessage.model_validate(message) for message in payload.get("messages", [])]

        findings = _scan_messages_for_secrets(messages)
        if findings and config.redaction_mode == "block":
            raise HTTPException(
                status_code=400,
                detail={"error": "redaction policy blocked request", "findings": sorted(set(findings))},
            )
        if findings:
            logger.warning(
                "request.redaction.findings requester=%s findings=%s",
                requester,
                sorted(set(findings)),
            )

        task_class = TaskClass(payload.get("task_class", TaskClass.CHAT.value))
        tool_authority = bool(payload.get("tool_authority_required", False))
        if _has_tools(payload):
            tool_authority = True
            if task_class == TaskClass.CHAT:
                task_class = TaskClass.TOOL_USE

        requested_model = payload.get("model")
        input_token_count = count_message_tokens(messages, model_id=requested_model)
        max_output = payload.get("max_tokens")

        request = InferenceRequest(
            requester=requester,
            sensitivity=Sensitivity(payload.get("sensitivity", Sensitivity.PUBLIC.value)),
            task_class=task_class,
            messages=messages,
            model=requested_model,
            max_output_tokens=max_output,
            stream=bool(payload.get("stream", False)),
            tool_authority_required=tool_authority,
            max_context_tokens=input_token_count + (max_output or 0),
            metadata={
                "raw_openai_payload": {k: v for k, v in payload.items() if k != "messages"},
                "redaction_findings": sorted(set(findings)) if findings else [],
                "counted_input_tokens": input_token_count,
            },
        )

        logger.info(
            "request.received request_id=%s requester=%s task=%s model=%s stream=%s tools=%s in_tokens=%d",
            request.id,
            request.requester,
            request.task_class.value,
            request.model,
            request.stream,
            tool_authority,
            input_token_count,
        )

        result = scheduler.route(request, store.list_nodes())
        if result.selected is None or not result.candidates:
            logger.warning(
                "request.no_candidates request_id=%s rejections=%s",
                request.id,
                [r.model_dump() for r in result.rejections],
            )
            raise HTTPException(status_code=503, detail=result.model_dump(mode="json"))

        logger.info(
            "request.route request_id=%s selected=%s/%s score=%.2f candidates=%d",
            request.id,
            result.selected.node.node_id,
            result.selected.model.id,
            result.selected.score,
            len(result.candidates),
        )

        if request.stream:
            return await _stream_with_fallback(
                request=request,
                payload=payload,
                candidates=result.candidates,
                store=store,
                config=config,
                input_token_count=input_token_count,
            )

        return await _execute_with_fallback(
            request=request,
            payload=payload,
            candidates=result.candidates,
            store=store,
            config=config,
            input_token_count=input_token_count,
        )

    return app


async def _execute_with_fallback(
    *,
    request: InferenceRequest,
    payload: dict[str, Any],
    candidates: list[RouteCandidate],
    store: MeshStore,
    config: MeshConfig,
    input_token_count: int,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for candidate in candidates:
        if (
            candidate.model.context_window
            and request.max_context_tokens
            and request.max_context_tokens > candidate.model.context_window
        ):
            attempts.append(
                {
                    "node_id": candidate.node.node_id,
                    "error": "requested context exceeds model context window",
                }
            )
            continue

        job = JobRecord(
            id=request.id if not attempts else f"{request.id}:r{len(attempts)}",
            requester=request.requester,
            node_id=candidate.node.node_id,
            model_id=candidate.model.id,
            backend_id=candidate.backend.id,
            sensitivity=request.sensitivity,
            task_class=request.task_class,
            status="started",
        )
        store.create_job(job)
        store.delta_node_jobs(candidate.node.node_id, 1)
        client = OpenAICompatClient(candidate.backend)
        started = time.perf_counter()
        try:
            response = await client.chat_completions(request, model_id=candidate.model.id, raw_payload=payload)
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            store.finish_job(job.id, status="error", error=str(exc), total_ms=elapsed_ms)
            store.delta_node_jobs(candidate.node.node_id, -1)
            attempts.append({"node_id": candidate.node.node_id, "error": str(exc)})
            logger.warning(
                "request.backend_error request_id=%s node=%s error=%s",
                request.id,
                candidate.node.node_id,
                exc,
            )
            if _is_transient_backend_error(exc) and candidate is not candidates[-1]:
                continue
            raise HTTPException(status_code=502, detail={"error": str(exc), "attempts": attempts}) from exc

        usage = response.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or input_token_count or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        total_ms = (time.perf_counter() - started) * 1000.0
        credit = _credit_cost(input_tokens, output_tokens, config.credits_per_1k_tokens)
        store.finish_job(
            job.id,
            status="completed",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_ms=total_ms,
        )
        store.add_ledger_entries(
            [
                UsageLedgerEntry(
                    job_id=job.id,
                    requester=request.requester,
                    node_id=candidate.node.node_id,
                    model_id=candidate.model.id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    gpu_ms_estimate=total_ms,
                    credit_cost=credit,
                )
            ]
        )
        store.delta_node_jobs(candidate.node.node_id, -1)
        logger.info(
            "request.completed request_id=%s node=%s in=%d out=%d ms=%.1f credit=%.4f attempts=%d",
            request.id,
            candidate.node.node_id,
            input_tokens,
            output_tokens,
            total_ms,
            credit,
            len(attempts) + 1,
        )
        response.setdefault("orchestro_mesh", {})["attempts"] = len(attempts) + 1
        response["orchestro_mesh"]["credit_cost"] = credit
        response["orchestro_mesh"]["request_id"] = request.id
        response["orchestro_mesh"]["node_id"] = candidate.node.node_id
        response["orchestro_mesh"]["model_id"] = candidate.model.id
        return response

    raise HTTPException(status_code=502, detail={"error": "all candidates failed", "attempts": attempts})


async def _stream_with_fallback(
    *,
    request: InferenceRequest,
    payload: dict[str, Any],
    candidates: list[RouteCandidate],
    store: MeshStore,
    config: MeshConfig,
    input_token_count: int,
) -> StreamingResponse:
    candidate = candidates[0]
    if (
        candidate.model.context_window
        and request.max_context_tokens
        and request.max_context_tokens > candidate.model.context_window
    ):
        raise HTTPException(
            status_code=400,
            detail={"error": "requested context exceeds model context window", "model_id": candidate.model.id},
        )

    job = JobRecord(
        id=request.id,
        requester=request.requester,
        node_id=candidate.node.node_id,
        model_id=candidate.model.id,
        backend_id=candidate.backend.id,
        sensitivity=request.sensitivity,
        task_class=request.task_class,
        status="started",
    )
    store.create_job(job)
    store.delta_node_jobs(candidate.node.node_id, 1)
    client = OpenAICompatClient(candidate.backend)
    started = time.perf_counter()
    counter = _StreamCounter()

    async def streamer():
        try:
            async for chunk in client.stream_chat_completions(request, model_id=candidate.model.id, raw_payload=payload):
                counter.feed(chunk)
                yield chunk
            output_tokens = count_text_tokens(counter.text, model_id=candidate.model.id)
            total_ms = (time.perf_counter() - started) * 1000.0
            credit = _credit_cost(input_token_count, output_tokens, config.credits_per_1k_tokens)
            store.finish_job(
                job.id,
                status="completed",
                input_tokens=input_token_count,
                output_tokens=output_tokens,
                total_ms=total_ms,
            )
            store.add_ledger_entries(
                [
                    UsageLedgerEntry(
                        job_id=job.id,
                        requester=request.requester,
                        node_id=candidate.node.node_id,
                        model_id=candidate.model.id,
                        input_tokens=input_token_count,
                        output_tokens=output_tokens,
                        gpu_ms_estimate=total_ms,
                        credit_cost=credit,
                    )
                ]
            )
            logger.info(
                "request.stream_completed request_id=%s node=%s in=%d out=%d ms=%.1f credit=%.4f",
                request.id,
                candidate.node.node_id,
                input_token_count,
                output_tokens,
                total_ms,
                credit,
            )
        except Exception as exc:  # noqa: BLE001
            store.finish_job(job.id, status="error", error=str(exc))
            raise
        finally:
            store.delta_node_jobs(candidate.node.node_id, -1)

    return StreamingResponse(
        streamer(),
        media_type="text/event-stream",
        headers={
            "X-Orchestro-Mesh-Request-Id": request.id,
            "X-Orchestro-Mesh-Node-Id": candidate.node.node_id,
            "X-Orchestro-Mesh-Model-Id": candidate.model.id,
        },
    )


def __getattr__(name: str):  # pragma: no cover - deferred module-level instantiation
    if name == "app":
        instance = create_gateway_app()
        globals()["app"] = instance
        return instance
    raise AttributeError(name)
