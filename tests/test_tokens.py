from orchestro_mesh.models import ChatMessage
from orchestro_mesh.tokens import count_message_tokens, count_text_tokens


def test_count_text_tokens_handles_empty():
    assert count_text_tokens("", model_id=None) == 0


def test_count_text_tokens_returns_positive_for_short_text():
    assert count_text_tokens("hello world", model_id=None) >= 1


def test_count_message_tokens_includes_per_message_overhead():
    short = [ChatMessage(role="user", content="a")]
    longer = [
        ChatMessage(role="system", content="you are helpful"),
        ChatMessage(role="user", content="explain orchestration in two sentences"),
    ]
    assert count_message_tokens(short) < count_message_tokens(longer)


def test_count_message_tokens_handles_structured_content():
    msg = ChatMessage(
        role="user",
        content=[{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": "..."}],
    )
    # Should not raise, and should count the text part.
    assert count_message_tokens([msg]) >= 1
