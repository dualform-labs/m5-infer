"""MMRS — Multi-Model Resident System.

Maintains a registry of simultaneously-resident backends so that routing to a
heavy/vision sub model becomes an O(1) pointer swap instead of a 10-20 s
load-from-disk cycle.

Activation policy
-----------------
MMRS is gated by the active runtime mode and remaining memory budget. On a
24 GB M5 base (moderate/aggressive/extreme) the second model simply will not
fit next to the 9B main + 2-bit shadow + KV cache + runtime, so the registry
degrades gracefully to the existing load/unload path in `SubModelController`.

On Pro (64-96 GB) and Ultra (128 GB+) tiers the registry eagerly pre-loads
whatever the `[engine.mmrs]` config block lists, and routing hits the
resident entry via `get_resident(role)`.

This module is intentionally *framework only* — it does not break the 24 GB
path. Wiring into the planner happens only when `is_active()` returns True.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ResidentEntry:
    role: str              # "main" | "heavy" | "vision" | "speed"
    model_path: str
    backend: object        # MLXTextBackend / MLXVisionBackend
    estimated_gb: float
    loaded_at: float


class MMRSRegistry:
    """Thread-safe registry of resident backends keyed by role."""

    def __init__(self, memory_budget_gb: float = 24.0):
        self._entries: dict[str, ResidentEntry] = {}
        self._lock = threading.RLock()
        self._budget_gb = memory_budget_gb
        self._enabled = False  # flipped on by activate() when memory allows

    def activate(self, total_resident_gb_budget: float) -> None:
        """Mark MMRS as active. Budget is the upper bound we may keep resident."""
        with self._lock:
            self._budget_gb = total_resident_gb_budget
            self._enabled = True
            logger.info("MMRS activated (budget=%.1f GB)", total_resident_gb_budget)

    def is_active(self) -> bool:
        return self._enabled

    def register(self, role: str, model_path: str, backend: object,
                 estimated_gb: float) -> None:
        with self._lock:
            # Capacity check — refuse if would blow the budget.
            current = sum(e.estimated_gb for e in self._entries.values())
            if current + estimated_gb > self._budget_gb:
                logger.warning(
                    "MMRS register rejected (role=%s gb=%.1f, %.1f/%.1f used)",
                    role, estimated_gb, current, self._budget_gb,
                )
                return
            self._entries[role] = ResidentEntry(
                role=role, model_path=model_path, backend=backend,
                estimated_gb=estimated_gb, loaded_at=time.time(),
            )
            logger.info("MMRS: %s resident (%.1f GB, total=%.1f/%.1f)",
                        role, estimated_gb,
                        current + estimated_gb, self._budget_gb)

    def get_resident(self, role: str) -> Optional[object]:
        with self._lock:
            entry = self._entries.get(role)
            return entry.backend if entry else None

    def evict(self, role: str) -> None:
        with self._lock:
            entry = self._entries.pop(role, None)
            if entry and hasattr(entry.backend, "unload_model"):
                try:
                    entry.backend.unload_model()
                except Exception:
                    logger.debug("MMRS evict unload failed", exc_info=True)

    def stats(self) -> dict:
        with self._lock:
            return {
                "enabled": self._enabled,
                "budget_gb": self._budget_gb,
                "resident": {r: e.estimated_gb for r, e in self._entries.items()},
                "total_gb": sum(e.estimated_gb for e in self._entries.values()),
            }


_global_registry: MMRSRegistry | None = None


def get_mmrs_registry() -> MMRSRegistry:
    global _global_registry
    if _global_registry is None:
        _global_registry = MMRSRegistry()
    return _global_registry
