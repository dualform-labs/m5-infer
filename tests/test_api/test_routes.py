"""Tests for API routes — basic functionality without real model."""
import pytest
from unittest.mock import MagicMock, patch
from app.api.schemas import ChatCompletionRequest, ChatMessage

def test_chat_completion_request_defaults():
    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="hello")]
    )
    assert req.model == "default"
    assert req.max_tokens == 4096
    assert req.temperature == 0.0
    assert req.stream is False
    assert req.session_id is None

def test_chat_completion_request_with_extensions():
    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="hello")],
        speed_priority="high",
        prefer_long_generation=True,
        session_id="test-123",
    )
    assert req.speed_priority == "high"
    assert req.prefer_long_generation is True
    assert req.session_id == "test-123"

def test_chat_message_variants():
    # User message
    msg = ChatMessage(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"

    # Assistant message with no content
    msg2 = ChatMessage(role="assistant")
    assert msg2.content is None

    # Tool message
    msg3 = ChatMessage(role="tool", content="result", tool_call_id="call_123")
    assert msg3.tool_call_id == "call_123"
