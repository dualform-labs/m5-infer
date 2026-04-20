"""X2: Dynamic Precision Cascading — Confidence-based precision routing.

Routes token generation to 2-bit or 4-bit based on prediction confidence.
High-confidence tokens (>threshold) use 2-bit for speed.
Verify-then-accept ensures lossless output.
"""

from __future__ import annotations
import mlx.core as mx
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class ConfidenceRouter:
    """Routes inference to 2-bit or 4-bit based on confidence."""

    def __init__(self, threshold: float | None = None):
        if threshold is None:
            settings = get_settings()
            threshold = settings.innovation.x2_dpc_confidence_threshold
        self._threshold = threshold
        self._stats = {"total": 0, "2bit_hits": 0, "4bit_fallbacks": 0}

    def should_use_2bit(self, logits: mx.array) -> bool:
        """Check if the last token's confidence exceeds threshold.

        Args:
            logits: Model output logits [vocab_size] or [1, vocab_size]

        Returns:
            True if top-1 probability exceeds threshold (use 2-bit next step)
        """
        # Handle [B, T, V] or [T, V] or [V] shapes — always use last token
        while logits.ndim > 1:
            logits = logits[-1]

        # Compute softmax for top-1 probability
        max_logit = mx.max(logits)
        log_sum_exp = mx.logsumexp(logits)
        top1_log_prob = max_logit - log_sum_exp
        top1_prob = mx.exp(top1_log_prob)

        self._stats["total"] += 1
        confident = top1_prob.item() >= self._threshold
        if confident:
            self._stats["2bit_hits"] += 1
        else:
            self._stats["4bit_fallbacks"] += 1

        return confident

    def get_2bit_hit_rate(self) -> float:
        if self._stats["total"] == 0:
            return 0.0
        return self._stats["2bit_hits"] / self._stats["total"]

    def reset_stats(self) -> None:
        self._stats = {"total": 0, "2bit_hits": 0, "4bit_fallbacks": 0}

    def get_stats(self) -> dict:
        return {
            **self._stats,
            "hit_rate": self.get_2bit_hit_rate(),
            "threshold": self._threshold,
        }
