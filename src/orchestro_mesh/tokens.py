from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from orchestro_mesh.models import ChatMessage

_PER_MESSAGE_OVERHEAD = 4


def _fallback(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _get_encoder(model_id: str | None) -> Any | None:
    try:
        import tiktoken  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        if model_id:
            return tiktoken.encoding_for_model(model_id)
    except (KeyError, ValueError):
        pass
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_text_tokens(text: str, model_id: str | None = None) -> int:
    if not text:
        return 0
    encoder = _get_encoder(model_id)
    if encoder is None:
        return _fallback(text)
    return len(encoder.encode(text))


def count_message_tokens(messages: Iterable[ChatMessage], model_id: str | None = None) -> int:
    encoder = _get_encoder(model_id)
    total = 0
    for message in messages:
        content = message.content
        if isinstance(content, str):
            total += len(encoder.encode(content)) if encoder else _fallback(content)
        elif isinstance(content, list):
            for part in content:
                text = part.get("text") if isinstance(part, dict) else None
                if isinstance(text, str):
                    total += len(encoder.encode(text)) if encoder else _fallback(text)
        total += _PER_MESSAGE_OVERHEAD
    return total
