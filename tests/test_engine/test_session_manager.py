"""Tests for session manager."""
import time
import pytest
from unittest.mock import patch
from app.engine.session_manager import SessionManager

def test_create_new_session():
    sm = SessionManager()
    session = sm.get_or_create()
    assert session.session_id is not None
    assert sm.count() == 1

def test_get_existing_session():
    sm = SessionManager()
    s1 = sm.get_or_create("test-id")
    s2 = sm.get_or_create("test-id")
    assert s1.session_id == s2.session_id
    assert sm.count() == 1

def test_append_message():
    sm = SessionManager()
    s = sm.get_or_create("test-id")
    sm.append_message("test-id", "user", "hello")
    sm.append_message("test-id", "assistant", "hi")
    msgs = sm.get_messages("test-id")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["content"] == "hi"

def test_append_to_nonexistent_raises():
    sm = SessionManager()
    with pytest.raises(ValueError):
        sm.append_message("nonexistent", "user", "hello")

def test_cleanup_expired():
    sm = SessionManager()
    s_old = sm.get_or_create("old-session")
    s_old.last_used_at = time.time() - 999999  # Very old
    s_new = sm.get_or_create("new-session")
    s_new.last_used_at = time.time()  # Ensure fresh
    assert sm.count() == 2

    with patch("app.engine.session_manager.get_settings") as mock_settings:
        mock_settings.return_value.engine.session_idle_ttl_sec = 300
        cleaned = sm.cleanup_expired()
    assert cleaned >= 1  # At least old-session removed
    assert "new-session" in sm.list_sessions()  # New session survives
