"""SQLite-based metrics persistence with async-safe batch commits.

Previous implementation called `commit()` on every log_request → a synchronous
disk fsync blocking the request-handling thread. For bursty workloads this
adds measurable latency.

This version buffers INSERTs in memory and flushes on one of:
  * buffer size reaches `_flush_every_rows` (default 20)
  * buffer age exceeds `_flush_every_sec` (default 5s)
  * explicit `flush()` / `close()`
"""

from __future__ import annotations
import sqlite3
import json
import time
import threading
from pathlib import Path
from app.core.logging import get_logger

logger = get_logger(__name__)


class SQLiteStore:
    """Persists inference metrics to SQLite with buffered commits."""

    def __init__(
        self,
        db_path: str = "state/metrics.db",
        flush_every_rows: int = 20,
        flush_every_sec: float = 5.0,
    ):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._flush_every_rows = flush_every_rows
        self._flush_every_sec = flush_every_sec
        self._buffer: list[tuple] = []
        self._last_flush_ts = time.time()
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        # check_same_thread=False: we guard with self._lock ourselves.
        # This lets the uvicorn worker share the connection safely.
        self._conn = sqlite3.connect(
            str(self._db_path), check_same_thread=False,
        )
        # Use WAL for better write concurrency (reader ≠ writer lock)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.Error:
            pass
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS request_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                request_id TEXT,
                session_id TEXT,
                model TEXT,
                prompt_tokens INTEGER,
                output_tokens INTEGER,
                ttft_ms REAL,
                decode_tok_per_sec REAL,
                total_latency_ms REAL,
                peak_memory_gb REAL,
                prefix_cache_hit INTEGER,
                speculative_used INTEGER,
                finish_reason TEXT,
                extra TEXT
            )
        """)
        self._conn.commit()

    def log_request(self, **kwargs) -> None:
        if self._conn is None:
            return
        extra = {k: v for k, v in kwargs.items() if k not in {
            "request_id", "session_id", "model", "prompt_tokens",
            "output_tokens", "ttft_ms", "decode_tok_per_sec",
            "total_latency_ms", "peak_memory_gb", "prefix_cache_hit",
            "speculative_used", "finish_reason",
        }}
        row = (
            time.time(),
            kwargs.get("request_id"),
            kwargs.get("session_id"),
            kwargs.get("model"),
            kwargs.get("prompt_tokens", 0),
            kwargs.get("output_tokens", 0),
            kwargs.get("ttft_ms", 0),
            kwargs.get("decode_tok_per_sec", 0),
            kwargs.get("total_latency_ms", 0),
            kwargs.get("peak_memory_gb", 0),
            int(kwargs.get("prefix_cache_hit", False)),
            int(kwargs.get("speculative_used", False)),
            kwargs.get("finish_reason"),
            json.dumps(extra) if extra else None,
        )
        with self._lock:
            self._buffer.append(row)
            if self._should_flush_locked():
                self._flush_locked()

    def _should_flush_locked(self) -> bool:
        if len(self._buffer) >= self._flush_every_rows:
            return True
        if (time.time() - self._last_flush_ts) >= self._flush_every_sec:
            return True
        return False

    def _flush_locked(self) -> None:
        if not self._buffer or self._conn is None:
            self._last_flush_ts = time.time()
            return
        try:
            self._conn.executemany(
                """INSERT INTO request_metrics
                   (timestamp, request_id, session_id, model, prompt_tokens,
                    output_tokens, ttft_ms, decode_tok_per_sec, total_latency_ms,
                    peak_memory_gb, prefix_cache_hit, speculative_used,
                    finish_reason, extra)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                self._buffer,
            )
            self._conn.commit()
            self._buffer.clear()
        except sqlite3.Error as e:
            logger.warning("SQLite flush failed (%d buffered rows): %s",
                           len(self._buffer), e)
            # Keep the buffer so we retry on next flush.
        self._last_flush_ts = time.time()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def get_recent(self, limit: int = 100) -> list[dict]:
        if self._conn is None:
            return []
        # Flush any pending so recent queries see the latest inserts.
        self.flush()
        cursor = self._conn.execute(
            "SELECT * FROM request_metrics ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def close(self) -> None:
        try:
            self.flush()
        finally:
            with self._lock:
                if self._conn:
                    self._conn.close()
                    self._conn = None
