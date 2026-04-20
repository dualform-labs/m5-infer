"""Background compiler — moves heavy analysis OFF critical path.

On CTRSP cache miss for a large prompt, v3 used to synchronously run
`analyze_prompt_redundancy` (scans all 20-grams over 12K tokens → 2-3 seconds
of Python-bound work) and MTAB observation (sha256 over 12K tokens →
additional ~50ms). Both blocked the first-token latency for no user benefit
on COLD tool_heavy calls.

This module dispatches those tasks to a dedicated low-priority background
thread. Results are logged when ready, and TPCEntry flags track completion
so we don't duplicate work.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CompileTask:
    name: str
    func: Callable[[], None]
    submitted_at: float


class BackgroundCompiler:
    """Single-threaded background worker for non-critical analysis.

    Single-threaded to avoid contending with GPU prefill for CPU time during
    active requests. Tasks run serially when the worker thread wakes up.
    """

    def __init__(self):
        self._queue: "queue.Queue[Optional[CompileTask]]" = queue.Queue(maxsize=64)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._stats = {
            "submitted": 0,
            "completed": 0,
            "errors": 0,
            "dropped_full": 0,
            "total_runtime_ms": 0.0,
        }

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._run, name="tpc-bg-compiler", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass

    def submit(self, name: str, func: Callable[[], None]) -> bool:
        """Submit a task. Returns False if queue is full (task dropped)."""
        if not self._running:
            self.start()
        task = CompileTask(name=name, func=func, submitted_at=time.time())
        try:
            self._queue.put_nowait(task)
            self._stats["submitted"] += 1
            return True
        except queue.Full:
            self._stats["dropped_full"] += 1
            return False

    def _run(self) -> None:
        while self._running:
            try:
                task = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if task is None:
                break
            t0 = time.time()
            try:
                task.func()
                self._stats["completed"] += 1
            except Exception as e:
                self._stats["errors"] += 1
                logger.warning("TPC background task '%s' failed: %s", task.name, e)
            finally:
                self._stats["total_runtime_ms"] += (time.time() - t0) * 1000.0

    def stats(self) -> dict:
        return dict(self._stats)


_global_compiler: Optional[BackgroundCompiler] = None
_lock = threading.Lock()


def get_background_compiler() -> BackgroundCompiler:
    global _global_compiler
    if _global_compiler is None:
        with _lock:
            if _global_compiler is None:
                _global_compiler = BackgroundCompiler()
                _global_compiler.start()
    return _global_compiler
