"""Watchdog for crash, stall, and memory-pressure detection."""

from __future__ import annotations

import asyncio
import time

from app.core.config import get_settings
from app.core.logging import get_logger
from app.engine.memory_guard import MemoryGuard
from app.engine.session_manager import SessionManager
from app.planner.plan_types import MemoryState

logger = get_logger(__name__)


class Supervisor:
    """Monitors system health and triggers actions on stall/OOM/session leaks.

    Unlike the previous iteration, the monitor loop is NOT cosmetic —
    stall detection actually fires a warning (cancelling in-flight generation
    is left to the caller via a flag), memory RED triggers cache_manager /
    CTRSP eviction, and session cleanup runs periodically regardless of
    request traffic.
    """

    def __init__(self, memory_guard: MemoryGuard, session_manager: SessionManager):
        self._memory_guard = memory_guard
        self._session_manager = session_manager
        self._running = False
        self._task: asyncio.Task | None = None
        self._consecutive_failures = 0
        self._last_generation_time = time.time()
        self._generation_in_progress = False
        self._stall_detected = False
        # Hooks set by api/server.py on startup — called when RED state
        # requires actual (not advisory) remediation.
        self._ctrsp_manager = None
        self._cache_manager = None
        self._stats = {
            "memory_red_count": 0,
            "session_cleanups": 0,
            "stall_detections": 0,
            "ctrsp_evictions_by_memory": 0,
        }

    def set_remediation_hooks(self, ctrsp_manager=None, cache_manager=None) -> None:
        """Wire actual remediation targets for memory RED state."""
        self._ctrsp_manager = ctrsp_manager
        self._cache_manager = cache_manager

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Supervisor started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Supervisor stopped")

    def report_generation_start(self) -> None:
        """Called at the start of a generation — enables stall detection."""
        self._last_generation_time = time.time()
        self._generation_in_progress = True
        self._stall_detected = False

    def report_generation_activity(self) -> None:
        """Called when generation produces a token (resets stall timer)."""
        self._last_generation_time = time.time()
        self._consecutive_failures = 0
        self._generation_in_progress = False

    def report_generation_failure(self) -> None:
        self._consecutive_failures += 1
        self._generation_in_progress = False

    def stall_detected(self) -> bool:
        """Caller can poll this to abort the in-flight generation."""
        return self._stall_detected

    def get_stats(self) -> dict:
        return dict(self._stats)

    async def _monitor_loop(self) -> None:
        settings = get_settings()
        check_interval = 10  # Was 30s; now 10s for quicker reaction

        while self._running:
            try:
                await asyncio.sleep(check_interval)

                # ── Memory check ─────────────────────────
                snapshot = self._memory_guard.check_memory()
                if snapshot.state == MemoryState.RED:
                    self._stats["memory_red_count"] += 1
                    self._memory_guard.apply_degradation(snapshot)

                    # Actually act on RED: drop CTRSP entries and prefix cache
                    # (previously these were only "recommended" strings).
                    if self._ctrsp_manager is not None:
                        try:
                            n = self._ctrsp_manager.clear()
                            if n > 0:
                                self._stats["ctrsp_evictions_by_memory"] += n
                                logger.warning(
                                    "Supervisor: RED → cleared %d CTRSP entries", n,
                                )
                        except Exception:
                            logger.exception("Failed to clear CTRSP under RED")

                    if self._cache_manager is not None:
                        try:
                            if hasattr(self._cache_manager, "clear_all"):
                                self._cache_manager.clear_all()
                        except Exception:
                            logger.exception("Failed to clear cache under RED")

                    # Also trigger mode auto-downgrade to prevent recurrence
                    try:
                        new_mode = self._memory_guard.auto_downgrade()
                        if new_mode:
                            logger.warning(
                                "Supervisor: mode downgraded to %s due to memory pressure",
                                new_mode,
                            )
                    except Exception:
                        logger.exception("auto_downgrade failed")

                    logger.warning(
                        "Memory RED: used=%.2f GB, available=%.2f GB",
                        snapshot.total_used_gb,
                        snapshot.available_gb,
                    )

                # ── Periodic session cleanup ────────────
                cleaned = self._session_manager.cleanup_expired()
                if cleaned > 0:
                    self._stats["session_cleanups"] += cleaned

                # ── Stall detection (was dead code) ─────
                # If a generation is in flight and we've heard no activity
                # for > stall_timeout_sec, flag the stall. The generator
                # itself checks `supervisor.stall_detected()` between tokens.
                if self._generation_in_progress:
                    idle = time.time() - self._last_generation_time
                    stall_sec = settings.engine.stall_timeout_sec
                    if idle > stall_sec and not self._stall_detected:
                        self._stall_detected = True
                        self._stats["stall_detections"] += 1
                        logger.error(
                            "Supervisor: STALL detected — no tokens for %.1fs "
                            "(threshold %ds). Generation will be aborted on next poll.",
                            idle, stall_sec,
                        )

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Supervisor monitor error")
