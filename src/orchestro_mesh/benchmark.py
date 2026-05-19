from __future__ import annotations

import time
from collections.abc import Sequence

from orchestro_mesh.models import BenchmarkSample, ChatMessage, InferenceRequest, TaskClass
from orchestro_mesh.openai_client import OpenAICompatClient

DEFAULT_BENCH_MESSAGES = [
    ChatMessage(role="system", content="You are a concise benchmark assistant."),
    ChatMessage(role="user", content="Write a compact checklist for validating a private inference mesh worker."),
]

TOOL_BENCH_MESSAGES = [
    ChatMessage(role="system", content="You are a tool-planning assistant. Do not execute tools."),
    ChatMessage(role="user", content="Given a failing pytest run, outline the tool calls and expected outputs needed to debug it."),
]


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


def extract_output_text(response: dict) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    return ""


async def run_chat_benchmark(
    client: OpenAICompatClient,
    model_id: str,
    messages: Sequence[ChatMessage] | None = None,
    prompt_id: str = "chat-default",
    task_class: TaskClass = TaskClass.CHAT,
) -> BenchmarkSample:
    bench_messages = list(messages or (TOOL_BENCH_MESSAGES if task_class == TaskClass.TOOL_USE else DEFAULT_BENCH_MESSAGES))
    input_tokens = sum(estimate_tokens(str(message.content or "")) for message in bench_messages)
    request = InferenceRequest(
        requester="benchmark",
        task_class=task_class,
        messages=bench_messages,
        model=model_id,
        max_output_tokens=256,
    )
    started = time.perf_counter()
    response = await client.chat_completions(request, model_id=model_id)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    output_text = extract_output_text(response)
    output_tokens = estimate_tokens(output_text)
    elapsed_s = max(elapsed_ms / 1000.0, 0.001)
    decode_tps = output_tokens / elapsed_s
    return BenchmarkSample(
        prompt_id=prompt_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_ms=elapsed_ms,
        prompt_tps=input_tokens / elapsed_s,
        decode_tps=decode_tps,
        end_to_end_tps=(input_tokens + output_tokens) / elapsed_s,
        tool_roundtrip_tps=decode_tps if task_class == TaskClass.TOOL_USE else None,
        measured_by="orchestro-mesh",
    )
