"""Traditional Speculative Decoding — draft model + verification.

Uses a separate small draft model to propose tokens, then the main
model verifies them in a single forward pass. Acceptance is based
on probability matching (rejection sampling).
"""

from __future__ import annotations
from dataclasses import dataclass
import mlx.core as mx
from typing import Iterator
from app.backend.adapter import GenerationChunk
from app.core.logging import get_logger

logger = get_logger(__name__)

@dataclass
class SpeculativeStats:
    total_draft_tokens: int = 0
    accepted_tokens: int = 0
    total_steps: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.total_draft_tokens


class SpeculativeDecoder:
    """Traditional speculative decoding with a separate draft model.

    Algorithm:
    1. Generate N tokens from draft model
    2. Verify all N tokens with main model in one forward pass
    3. Accept prefix of tokens matching acceptance criterion
    4. Reject rest and rewind caches
    """

    def __init__(self, num_draft_tokens: int = 4):
        self._num_draft = num_draft_tokens
        self._stats = SpeculativeStats()

    def decode_step(
        self,
        main_model,
        draft_model,
        prompt: mx.array,
        main_cache=None,
        draft_cache=None,
    ) -> list[int]:
        """Run one speculative decode step.

        Returns list of accepted token IDs.
        Full implementation depends on mlx_lm's speculative_generate_step.
        This is a placeholder that will be wired up in integration.
        """
        # Phase 5 placeholder — will use mlx_lm.generate.speculative_generate_step
        logger.debug("Speculative decode step (placeholder)")
        self._stats.total_steps += 1
        return []

    def get_stats(self) -> dict:
        return {
            "total_draft": self._stats.total_draft_tokens,
            "accepted": self._stats.accepted_tokens,
            "acceptance_rate": self._stats.acceptance_rate,
            "steps": self._stats.total_steps,
            "num_draft_tokens": self._num_draft,
        }

    def reset_stats(self) -> None:
        self._stats = SpeculativeStats()
