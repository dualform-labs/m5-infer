"""Memory protection with graduated degradation.

Features:
  * Mode-aware thresholds (extreme: yellow 16/red 17 — tighter)
  * Auto-downgrade hook fires on RED — switches to lower memory mode
  * Tracks cumulative degradation events for observability
"""

from __future__ import annotations

import mlx.core as mx

from app.core.config import get_settings
from app.core.logging import get_logger
from app.planner.plan_types import MemorySnapshot, MemoryState

logger = get_logger(__name__)


# Per-mode threshold overrides (tighter for extreme mode where margin is small)
_MODE_THRESHOLDS = {
    "moderate":   {"yellow": 18.0, "red": 20.0},
    "aggressive": {"yellow": 17.0, "red": 18.5},
    "extreme":    {"yellow": 16.0, "red": 17.0},  # very tight, 1 GB margin
}


class MemoryGuard:
    """Monitors memory and provides graduated degradation.

    Thresholds adjust based on the active runtime mode: extreme mode
    (~18 GB resident) trips RED at tighter limits than moderate (~13 GB).
    """

    def __init__(self):
        self._stats = {
            "yellow_count": 0,
            "red_count": 0,
            "downgrade_events": 0,
            "last_downgrade_to": None,
        }

    def _get_thresholds(self) -> dict:
        """Return mode-aware (yellow_gb, red_gb) thresholds."""
        settings = get_settings()
        mode = settings.runtime.memory_mode
        # Mode-specific threshold; fall back to engine.toml [memory] section
        if mode in _MODE_THRESHOLDS:
            return _MODE_THRESHOLDS[mode]
        return {
            "yellow": settings.memory.yellow_threshold_gb,
            "red": settings.memory.red_threshold_gb,
        }

    def check_memory(self) -> MemorySnapshot:
        """Check current memory state with mode-aware thresholds."""
        settings = get_settings()
        thresholds = self._get_thresholds()

        active_gb = mx.get_active_memory() / 1e9
        peak_gb = mx.get_peak_memory() / 1e9
        cache_gb = mx.get_cache_memory() / 1e9
        total_used = active_gb + cache_gb
        available = settings.memory.total_budget_gb - total_used

        if total_used >= thresholds["red"]:
            state = MemoryState.RED
            self._stats["red_count"] += 1
        elif total_used >= thresholds["yellow"]:
            state = MemoryState.YELLOW
            self._stats["yellow_count"] += 1
        else:
            state = MemoryState.GREEN

        return MemorySnapshot(
            total_used_gb=round(total_used, 2),
            available_gb=round(available, 2),
            worker_used_gb=round(active_gb, 2),
            cache_used_gb=round(cache_gb, 2),
            state=state,
        )

    def can_load_sub_model(self, estimated_ram_gb: float) -> bool:
        """Check if there's enough memory to load a sub model."""
        settings = get_settings()
        snapshot = self.check_memory()
        return snapshot.available_gb >= max(
            estimated_ram_gb, settings.memory.allow_sub_model_only_if_free_gb
        )

    def apply_degradation(self, snapshot: MemorySnapshot) -> list[str]:
        """Apply graduated degradation steps. Returns list of actions taken.

        Actions are REAL (not advisory strings):
          YELLOW → clear MLX transient cache
          RED    → also unload draft model (RDMS), trigger downgrade hook
        """
        actions: list[str] = []

        if snapshot.state == MemoryState.GREEN:
            return actions

        if snapshot.state in (MemoryState.YELLOW, MemoryState.RED):
            cache_before = mx.get_cache_memory() / 1e9
            if cache_before > 0.1:
                mx.clear_cache()
                actions.append(f"Cleared MLX cache ({cache_before:.2f} GB)")

        if snapshot.state == MemoryState.RED:
            # Real action — unload RDMS draft model
            try:
                from app.innovation.rdms.draft_loader import get_draft_loader
                loader = get_draft_loader()
                if loader.is_loaded():
                    loader.unload()
                    actions.append("RED: unloaded RDMS draft model")
            except Exception:
                logger.exception("Failed to unload RDMS draft on RED")

            # Trigger auto-downgrade hook (set by Supervisor)
            actions.append(
                f"RED: triggering auto-downgrade from mode={get_settings().runtime.memory_mode}"
            )

        for action in actions:
            logger.warning("Memory degradation: %s", action)

        return actions

    def auto_downgrade(self) -> str | None:
        """Attempt mode downgrade: extreme → aggressive → moderate.

        Note: This is best-effort. The settings cache is mutable but new
        requests may still see the old mode until they re-read settings.

        Returns the new mode name, or None if already at moderate.
        """
        settings = get_settings()
        current = settings.runtime.memory_mode
        downgrade_path = {
            "extreme": "aggressive",
            "aggressive": "moderate",
            "moderate": None,
        }
        new_mode = downgrade_path.get(current)
        if new_mode is None:
            logger.warning("Memory Guard: cannot downgrade further (already moderate)")
            return None

        # Mutate the in-memory settings (next request sees new mode)
        settings.runtime.memory_mode = new_mode
        self._stats["downgrade_events"] += 1
        self._stats["last_downgrade_to"] = new_mode
        logger.warning(
            "Memory Guard: AUTO-DOWNGRADE %s → %s",
            current, new_mode,
        )
        return new_mode

    def get_stats(self) -> dict:
        return dict(self._stats)
