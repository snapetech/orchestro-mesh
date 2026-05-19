from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from orchestro_mesh.config import MeshConfig
from orchestro_mesh.models import ChatMessage, InferenceRequest, JobRecord, Sensitivity, TaskClass
from orchestro_mesh.openai_client import OpenAICompatClient
from orchestro_mesh.scheduler import Scheduler
from orchestro_mesh.store import MeshStore


def create_gateway_app(config: MeshConfig | None = None) -> FastAPI:
    config = config or MeshConfig()
    store = MeshStore(config.store_path)
    for node in config.nodes:
        store.upsert_node(node)
    scheduler = Scheduler(local_node_id=config.local_node_id)

    app = FastAPI(title="Orchestro Mesh Gateway", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "nodes": len(store.list_nodes())}

    @app.get("/mesh/nodes")
    async def list_nodes() -> list[dict[str, Any]]:
        return [node.model_dump(mode="json") for node in store.list_nodes()]

    @app.post("/mesh/nodes")
    async def register_node(node: dict[str, Any]) -> dict[str, Any]:
        from orchestro_mesh.models import NodeInventory

        inventory = NodeInventory.model_validate(node)
        store.upsert_node(inventory)
        return {"ok": True, "node_id": inventory.node_id}

    @app.post("/mesh/route")
    async def route(request: InferenceRequest) -> dict[str, Any]:
        result = scheduler.route(request, store.list_nodes())
        return result.model_dump(mode="json")

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: dict[str, Any]) -> dict[str, Any]:
        messages = [ChatMessage.model_validate(message) for message in payload.get("messages", [])]
        request = InferenceRequest(
            requester=str(payload.get("user") or payload.get("requester") or "anonymous"),
            sensitivity=Sensitivity(payload.get("sensitivity", Sensitivity.PUBLIC.value)),
            task_class=TaskClass(payload.get("task_class", TaskClass.CHAT.value)),
            messages=messages,
            model=payload.get("model"),
            max_output_tokens=payload.get("max_tokens"),
            stream=bool(payload.get("stream", False)),
            metadata={"raw_openai_payload": {k: v for k, v in payload.items() if k not in {"messages"}}},
        )
        result = scheduler.route(request, store.list_nodes())
        if result.selected is None:
            raise HTTPException(status_code=503, detail=result.model_dump(mode="json"))
        selected = result.selected
        job = JobRecord(
            id=request.id,
            requester=request.requester,
            node_id=selected.node.node_id,
            model_id=selected.model.id,
            backend_id=selected.backend.id,
            sensitivity=request.sensitivity,
            task_class=request.task_class,
            status="started",
        )
        store.create_job(job)
        client = OpenAICompatClient(selected.backend)
        return await client.chat_completions(request, model_id=selected.model.id)

    return app


app = create_gateway_app()
