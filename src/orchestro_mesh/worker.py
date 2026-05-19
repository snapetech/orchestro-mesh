from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from orchestro_mesh.models import NodeInventory
from orchestro_mesh.openai_client import OpenAICompatClient


def create_worker_app(inventory: NodeInventory | None = None) -> FastAPI:
    app = FastAPI(title="Orchestro Mesh Worker", version="0.1.0")
    state: dict[str, Any] = {"inventory": inventory}

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": state["inventory"] is not None}

    @app.get("/inventory")
    async def get_inventory() -> dict[str, Any]:
        if state["inventory"] is None:
            raise HTTPException(status_code=503, detail="worker inventory is not configured")
        return state["inventory"].model_dump(mode="json")

    @app.post("/inventory")
    async def set_inventory(payload: dict[str, Any]) -> dict[str, Any]:
        state["inventory"] = NodeInventory.model_validate(payload)
        return {"ok": True, "node_id": state["inventory"].node_id}

    @app.post("/infer")
    async def infer(payload: dict[str, Any]) -> dict[str, Any]:
        from orchestro_mesh.models import InferenceRequest

        inventory = state["inventory"]
        if inventory is None:
            raise HTTPException(status_code=503, detail="worker inventory is not configured")
        request = InferenceRequest.model_validate(payload["request"])
        model_id = payload["model_id"]
        model = inventory.model_by_id(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail=f"model not found: {model_id}")
        backend = inventory.backend_by_id(model.backend_id)
        if backend is None:
            raise HTTPException(status_code=404, detail=f"backend not found: {model.backend_id}")
        client = OpenAICompatClient(backend)
        return await client.chat_completions(request, model_id=model.id)

    return app


app = create_worker_app()
