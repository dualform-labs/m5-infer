"""N5: Expert Route Prediction — Fallback to standard router."""

from __future__ import annotations
from app.core.logging import get_logger

logger = get_logger(__name__)

class RouterFallback:
    """Manages fallback from predicted routing to standard router.

    When prediction is unavailable or wrong, falls back to the model's
    native MoE router. Cost of fallback = one extra router computation.
    """

    def __init__(self):
        self._fallback_count = 0
        self._total_count = 0

    def should_fallback(self, prediction: list[int] | None) -> bool:
        """Check if we should fall back to standard routing."""
        self._total_count += 1
        if prediction is None:
            self._fallback_count += 1
            return True
        return False

    def record_forced_fallback(self) -> None:
        """Record when prediction was wrong and we had to fallback."""
        self._fallback_count += 1

    def get_fallback_rate(self) -> float:
        if self._total_count == 0:
            return 0.0
        return self._fallback_count / self._total_count

    def get_stats(self) -> dict:
        return {
            "fallbacks": self._fallback_count,
            "total": self._total_count,
            "fallback_rate": self.get_fallback_rate(),
        }
