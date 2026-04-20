"""X2: Dynamic Precision Cascading — Verify-then-accept for lossless guarantee.

After 2-bit forward pass, verify that 4-bit would produce the same top-1 token.
If not, fall back to 4-bit result. This ensures output is identical to pure 4-bit.

NOTE: Lossless guarantee only holds for greedy decoding (temperature=0).
When sampling (temperature>0), DPC should be disabled because the full logit
distribution from 2-bit weights differs from 4-bit, affecting sampling outcomes.
The confidence_router should check temperature before enabling 2-bit routing.
"""

from __future__ import annotations
import mlx.core as mx
from app.core.logging import get_logger

logger = get_logger(__name__)

class VerifyAcceptor:
    """Ensures DPC output matches 4-bit reference."""

    def __init__(self):
        self._stats = {"verified": 0, "accepted": 0, "rejected": 0}

    def verify_and_accept(
        self,
        logits_2bit: mx.array,
        logits_4bit: mx.array,
    ) -> tuple[mx.array, bool]:
        """Verify that 2-bit top-1 matches 4-bit top-1.

        Args:
            logits_2bit: Logits from 2-bit forward pass
            logits_4bit: Logits from 4-bit forward pass (or partial)

        Returns:
            (accepted_logits, was_2bit_accepted)
        """
        if logits_2bit.ndim > 1:
            logits_2bit = logits_2bit[-1]
        if logits_4bit.ndim > 1:
            logits_4bit = logits_4bit[-1]

        top1_2bit = mx.argmax(logits_2bit)
        top1_4bit = mx.argmax(logits_4bit)

        self._stats["verified"] += 1

        if top1_2bit.item() == top1_4bit.item():
            self._stats["accepted"] += 1
            return logits_2bit, True
        else:
            self._stats["rejected"] += 1
            return logits_4bit, False

    def get_acceptance_rate(self) -> float:
        if self._stats["verified"] == 0:
            return 0.0
        return self._stats["accepted"] / self._stats["verified"]

    def get_stats(self) -> dict:
        return {
            **self._stats,
            "acceptance_rate": self.get_acceptance_rate(),
        }
