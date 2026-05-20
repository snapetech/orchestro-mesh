from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException

from orchestro_mesh.auth import make_shared_token_dep
from orchestro_mesh.models import (
    InferenceRequest,
    NodeInventory,
    NodeStatus,
    Sensitivity,
    TaskClass,
)
from orchestro_mesh.openai_client import OpenAICompatClient
from orchestro_mesh.policy import evaluate_policy
from orchestro_mesh.worker_heartbeat import HeartbeatLoop

logger = logging.getLogger(__name__)


def create_worker_app(
    inventory: NodeInventory | None = None,
    worker_token: str | None = None,
    gateway_url: str | None = None,
    gateway_token: str | None = None,
    heartbeat_interval_s: float | None = None,
    shutdown_grace_s: float = 30.0,
) -> FastAPI:
    if worker_token is None:
        worker_token = os.environ.get("ORCHESTRO_MESH_WORKER_TOKEN")
    if gateway_url is None:
        gateway_url = os.environ.get("ORCHESTRO_MESH_GATEWAY_URL")
    if gateway_token is None:
        gateway_token = os.environ.get("ORCHESTRO_MESH_TOKEN")
    if heartbeat_interval_s is None:
        env_interval = os.environ.get("ORCHESTRO_MESH_HEARTBEAT_INTERVAL_S")
        heartbeat_interval_s = float(env_interval) if env_interval else 30.0

    auth_dep = make_shared_token_dep(worker_token)
    state: dict[str, Any] = {
        "inventory": inventory,
        "heartbeat_loop": None,
        "in_flight": 0,
        "in_flight_lock": asyncio.Lock(),
        "draining": False,
    }

    def _set_in_flight(delta: int) -> int:
        state["in_flight"] = max(0, state["in_flight"] + delta)
        inv = state["inventory"]
        if inv is not None:
            inv.current_jobs = state["in_flight"]
        return state["in_flight"]

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        loop: HeartbeatLoop | None = None
        if gateway_url:
            loop = HeartbeatLoop(
                gateway_url=gateway_url,
                token=gateway_token,
                inventory_provider=lambda: state["inventory"],
                interval_s=heartbeat_interval_s,
            )
            loop.start()
            state["heartbeat_loop"] = loop
        try:
            yield
        finally:
            state["draining"] = True
            inv = state["inventory"]
            if inv is not None:
                inv.status = NodeStatus.DRAINING
            # let heartbeat ship the DRAINING status one last time
            await asyncio.sleep(0.1)
            # wait for in-flight requests to finish
            deadline = asyncio.get_event_loop().time() + shutdown_grace_s
            while state["in_flight"] > 0 and asyncio.get_event_loop().time() < deadline:
                logger.info("worker.draining in_flight=%d", state["in_flight"])
                await asyncio.sleep(0.5)
            if state["in_flight"] > 0:
                logger.warning("worker.shutdown_force in_flight=%d", state["in_flight"])
            if loop is not None:
                await loop.stop()

    app = FastAPI(title="Orchestro Mesh Worker", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        inv = state["inventory"]
        return {
            "ok": inv is not None and not state["draining"],
            "draining": state["draining"],
            "in_flight": state["in_flight"],
            "heartbeat_active": state["heartbeat_loop"] is not None,
            "node_id": inv.node_id if inv else None,
        }

    @app.get("/inventory")
    async def get_inventory(_: None = Depends(auth_dep)) -> dict[str, Any]:
        if state["inventory"] is None:
            raise HTTPException(status_code=503, detail="worker inventory is not configured")
        return state["inventory"].model_dump(mode="json")

    @app.post("/inventory")
    async def set_inventory(
        payload: dict[str, Any],
        _: None = Depends(auth_dep),
    ) -> dict[str, Any]:
        state["inventory"] = NodeInventory.model_validate(payload)
        state["inventory"].current_jobs = state["in_flight"]
        return {"ok": True, "node_id": state["inventory"].node_id}

    @app.post("/drain")
    async def drain(_: None = Depends(auth_dep)) -> dict[str, Any]:
        state["draining"] = True
        inv = state["inventory"]
        if inv is not None:
            inv.status = NodeStatus.DRAINING
        return {"ok": True, "draining": True, "in_flight": state["in_flight"]}

    @app.post("/resume")
    async def resume(_: None = Depends(auth_dep)) -> dict[str, Any]:
        state["draining"] = False
        inv = state["inventory"]
        if inv is not None:
            inv.status = NodeStatus.ONLINE
        return {"ok": True, "draining": False}

    @app.post("/infer")
    async def infer(
        payload: dict[str, Any],
        _: None = Depends(auth_dep),
    ) -> dict[str, Any]:
        inventory = state["inventory"]
        if inventory is None:
            raise HTTPException(status_code=503, detail="worker inventory is not configured")
        if state["draining"]:
            raise HTTPException(status_code=503, detail="worker is draining")

        try:
            request = InferenceRequest.model_validate(payload["request"])
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid request payload: {exc}") from exc
        model_id = payload.get("model_id")
        if not model_id:
            raise HTTPException(status_code=400, detail="model_id is required")

        model = inventory.model_by_id(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail=f"model not found: {model_id}")
        backend = inventory.backend_by_id(model.backend_id)
        if backend is None:
            raise HTTPException(status_code=404, detail=f"backend not found: {model.backend_id}")

        # Defense-in-depth: even if the gateway approved this request, re-check our own policy.
        policy = evaluate_policy(request, inventory, model)
        if not policy.allowed:
            logger.warning(
                "worker.policy_block node=%s reason=%s sensitivity=%s task=%s",
                inventory.node_id,
                policy.reason,
                request.sensitivity.value if isinstance(request.sensitivity, Sensitivity) else request.sensitivity,
                request.task_class.value if isinstance(request.task_class, TaskClass) else request.task_class,
            )
            raise HTTPException(status_code=403, detail=f"worker policy denied: {policy.reason}")

        if inventory.policy.max_concurrent_jobs and state["in_flight"] >= inventory.policy.max_concurrent_jobs:
            raise HTTPException(status_code=429, detail="worker concurrency limit reached")

        async with state["in_flight_lock"]:
            _set_in_flight(+1)
        try:
            client = OpenAICompatClient(backend)
            return await client.chat_completions(request, model_id=model.id)
        finally:
            async with state["in_flight_lock"]:
                _set_in_flight(-1)

    return app


def __getattr__(name: str):  # pragma: no cover - deferred module-level instantiation
    if name == "app":
        instance = create_worker_app()
        globals()["app"] = instance
        return instance
    raise AttributeError(name)
