from __future__ import annotations

import time
from typing import Any

import httpx

from orchestro_mesh.models import BackendEndpoint, InferenceRequest


class OpenAICompatClient:
    def __init__(self, backend: BackendEndpoint) -> None:
        self.backend = backend

    async def chat_completions(self, request: InferenceRequest, model_id: str) -> dict[str, Any]:
        url = str(self.backend.base_url).rstrip("/") + "/chat/completions"
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": [message.model_dump(exclude_none=True) for message in request.messages],
            "stream": False,
        }
        if request.max_output_tokens:
            payload["max_tokens"] = request.max_output_tokens
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.backend.timeout_s, headers=self.backend.headers) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        data.setdefault("orchestro_mesh", {})["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return data
