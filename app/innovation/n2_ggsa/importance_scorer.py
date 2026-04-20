"""N2: GGSA — GatedDeltaNet state-based attention importance scoring.

Uses GatedDeltaNet recurrent state to predict which KV positions
are important for full attention layers. Qwen3.5 specific.

Research phase — effect measured at only ~1.05x, low priority.
"""

from __future__ import annotations
import mlx.core as mx
from app.core.logging import get_logger

logger = get_logger(__name__)

class ImportanceScorer:
    """Scores KV position importance using GatedDeltaNet state."""

    def __init__(self, top_k_ratio: float = 0.25):
        self._top_k_ratio = top_k_ratio
        self._ready = False

    def score_positions(self, gdn_state: mx.array, num_positions: int) -> mx.array:
        """Score KV positions by importance derived from GDN state.

        Args:
            gdn_state: [num_value_heads, value_dim, key_dim] recurrent state
            num_positions: Number of KV positions to score

        Returns:
            [num_positions] importance scores (higher = more important)
        """
        # Research: derive importance from state magnitude
        # Each position's importance approximated by its contribution to state
        state_norm = mx.sqrt(mx.sum(gdn_state * gdn_state, axis=(-2, -1)))
        # Broadcast to num_positions (simple uniform for now)
        scores = mx.ones(num_positions) * mx.mean(state_norm)
        return scores

    def get_top_k_mask(self, scores: mx.array) -> mx.array:
        """Create a boolean mask selecting top-k positions."""
        k = max(1, int(len(scores) * self._top_k_ratio))
        threshold = mx.sort(scores)[-k]
        return scores >= threshold

    def get_stats(self) -> dict:
        return {"top_k_ratio": self._top_k_ratio, "ready": self._ready}
