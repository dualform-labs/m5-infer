"""N3: SSEE acceptance rate monitor with auto-disable."""

from __future__ import annotations
from app.innovation.n3_ssee.self_speculative import SelfSpeculativeDecoder
from app.core.logging import get_logger

logger = get_logger(__name__)

class AcceptanceMonitor:
    """Monitors SSEE acceptance rate and auto-disables if too low."""

    def __init__(self, decoder: SelfSpeculativeDecoder, check_interval: int = 100):
        self._decoder = decoder
        self._check_interval = check_interval
        self._checks_done = 0

    def check(self) -> None:
        """Periodic check — call after each generation step."""
        stats = self._decoder.get_stats()
        total = stats["total_draft"]
        if total > 0 and total % self._check_interval == 0:
            self._checks_done += 1
            if self._decoder.should_auto_disable():
                self._decoder.disable()
                logger.warning(
                    "N3: Auto-disabled SSEE after %d tokens (rate=%.2f)",
                    total, stats["acceptance_rate"],
                )
