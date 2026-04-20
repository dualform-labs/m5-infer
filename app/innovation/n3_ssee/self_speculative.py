"""N3: Self-Speculative Early Exit — Self-speculative decode loop.

Uses first 4 layers as draft, full 32 layers for verification.
No extra model memory needed.
"""

from __future__ import annotations
from dataclasses import dataclass
import mlx.core as mx
from app.innovation.n3_ssee.exit_head import EarlyExitHead
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

@dataclass
class SSEEStats:
    """Statistics for self-speculative decoding."""
    total_draft_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    total_steps: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.total_draft_tokens


class SelfSpeculativeDecoder:
    """Self-speculative decoding using early exit as draft."""

    def __init__(self, exit_head: EarlyExitHead, num_draft_tokens: int | None = None):
        self._exit_head = exit_head
        if num_draft_tokens is None:
            settings = get_settings()
            num_draft_tokens = settings.innovation.n3_ssee_num_draft_tokens
        self._num_draft = num_draft_tokens
        self._stats = SSEEStats()
        self._enabled = True

    def is_enabled(self) -> bool:
        return self._enabled and self._exit_head.is_ready()

    def should_auto_disable(self) -> bool:
        """Check if acceptance rate is too low to be worthwhile."""
        settings = get_settings()
        min_rate = settings.innovation.n3_ssee_min_acceptance_rate
        if self._stats.total_draft_tokens < 100:
            return False  # Not enough data
        return self._stats.acceptance_rate < min_rate

    def disable(self) -> None:
        self._enabled = False
        logger.info(
            "N3: SSEE disabled (acceptance_rate=%.2f, below threshold)",
            self._stats.acceptance_rate,
        )

    def get_stats(self) -> dict:
        return {
            "enabled": self._enabled,
            "total_draft": self._stats.total_draft_tokens,
            "accepted": self._stats.accepted_tokens,
            "rejected": self._stats.rejected_tokens,
            "acceptance_rate": self._stats.acceptance_rate,
            "steps": self._stats.total_steps,
            "num_draft_tokens": self._num_draft,
        }
