"""In-memory inference session manager with TTL-based cleanup."""

import time
import uuid

from app.planner.plan_types import SessionState
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class SessionManager:
    """In-memory session storage with TTL-based cleanup."""

    def __init__(self):
        self._sessions: dict[str, SessionState] = {}

    def get_or_create(self, session_id: str | None = None) -> SessionState:
        """Get existing session or create a new one."""
        # Eager cleanup if many sessions accumulated
        if len(self._sessions) > 20:
            self.cleanup_expired()

        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            session.last_used_at = time.time()
            return session

        new_id = session_id or str(uuid.uuid4())
        session = SessionState(session_id=new_id)
        self._sessions[new_id] = session
        # T2 CPP — session lifecycle logs moved to DEBUG (fires per request).
        logger.debug(f"Created new session: {new_id}")
        return session

    def get(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def append_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to the session history."""
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        session.raw_messages.append({"role": role, "content": content})
        session.last_used_at = time.time()

    def get_messages(self, session_id: str) -> list[dict]:
        """Get all messages for a session."""
        session = self._sessions.get(session_id)
        if session is None:
            return []
        return session.raw_messages

    def cleanup_expired(self) -> int:
        """Remove sessions that have been idle beyond the TTL.

        Returns count of removed sessions.
        """
        settings = get_settings()
        ttl = settings.engine.session_idle_ttl_sec
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.last_used_at > ttl
        ]
        for sid in expired:
            del self._sessions[sid]
            logger.debug(f"Expired session: {sid}")
        return len(expired)

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    def count(self) -> int:
        return len(self._sessions)
