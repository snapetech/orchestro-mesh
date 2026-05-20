from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from orchestro_mesh.models import BackendEndpoint, BackendKind, InferenceRequest

PASSTHROUGH_PARAMS: frozenset[str] = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "n",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "response_format",
        "seed",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "user",
        "min_p",
        "repetition_penalty",
    }
)


def build_payload(
    request: InferenceRequest,
    model_id: str,
    raw_payload: dict[str, Any] | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [message.model_dump(exclude_none=True) for message in request.messages],
        "stream": stream,
    }
    if request.max_output_tokens:
        payload["max_tokens"] = request.max_output_tokens
    if raw_payload:
        for key in PASSTHROUGH_PARAMS:
            if key in raw_payload and raw_payload[key] is not None:
                payload[key] = raw_payload[key]
    return payload


def _mock_response(payload: dict[str, Any]) -> dict[str, Any]:
    text = "[mock-response]"
    prompt_tokens = sum(len(str(m.get("content") or "")) for m in payload.get("messages", [])) // 4
    completion_tokens = max(1, len(text) // 4)
    return {
        "id": "mock-completion",
        "object": "chat.completion",
        "model": payload.get("model"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class OpenAICompatClient:
    def __init__(self, backend: BackendEndpoint) -> None:
        self.backend = backend

    async def chat_completions(
        self,
        request: InferenceRequest,
        model_id: str,
        raw_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = build_payload(request, model_id, raw_payload=raw_payload, stream=False)
        started = time.perf_counter()
        if self.backend.kind == BackendKind.MOCK:
            data = _mock_response(payload)
            data.setdefault("orchestro_mesh", {})["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
            return data
        url = str(self.backend.base_url).rstrip("/") + "/chat/completions"
        async with httpx.AsyncClient(timeout=self.backend.timeout_s, headers=self.backend.headers) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        data.setdefault("orchestro_mesh", {})["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return data

    async def stream_chat_completions(
        self,
        request: InferenceRequest,
        model_id: str,
        raw_payload: dict[str, Any] | None = None,
    ) -> AsyncIterator[bytes]:
        payload = build_payload(request, model_id, raw_payload=raw_payload, stream=True)
        if self.backend.kind == BackendKind.MOCK:
            mock = _mock_response(payload)
            import json as _json

            yield f"data: {_json.dumps(mock)}\n\n".encode()
            yield b"data: [DONE]\n\n"
            return
        url = str(self.backend.base_url).rstrip("/") + "/chat/completions"
        client = httpx.AsyncClient(timeout=self.backend.timeout_s, headers=self.backend.headers)
        try:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk
        finally:
            await client.aclose()
