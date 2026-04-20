"""N5: Expert Route Prediction — Routing pattern profiler.

Records which experts are selected for different token types
to build a predictive routing model.
"""

from __future__ import annotations
from collections import defaultdict
from app.core.logging import get_logger

logger = get_logger(__name__)

class RouteProfiler:
    """Profiles MoE expert routing patterns by token context."""

    def __init__(self):
        # Map: context_key -> list of expert index sets
        self._patterns: dict[str, list[list[int]]] = defaultdict(list)
        self._total_observations = 0

    def observe(self, token_id: int, prev_token_id: int | None, selected_experts: list[int]) -> None:
        """Record a routing observation."""
        key = self._make_key(token_id, prev_token_id)
        self._patterns[key].append(selected_experts)
        self._total_observations += 1

    def predict(self, token_id: int, prev_token_id: int | None) -> list[int] | None:
        """Predict which experts will be selected based on past patterns.

        Returns None if no prediction available (insufficient data).
        """
        key = self._make_key(token_id, prev_token_id)
        observations = self._patterns.get(key)
        if not observations or len(observations) < 3:
            return None

        # Return the most recent pattern (simple heuristic)
        # Could be improved with frequency-based voting
        return observations[-1]

    def get_prediction_confidence(self, token_id: int, prev_token_id: int | None) -> float:
        """Estimate confidence in the prediction (0-1)."""
        key = self._make_key(token_id, prev_token_id)
        observations = self._patterns.get(key)
        if not observations or len(observations) < 3:
            return 0.0

        # Check consistency: how often the same pattern repeats
        last = observations[-1]
        matches = sum(1 for obs in observations[-10:] if obs == last)
        return matches / min(len(observations), 10)

    def _make_key(self, token_id: int, prev_token_id: int | None) -> str:
        return f"{prev_token_id or 0}:{token_id}"

    def get_stats(self) -> dict:
        return {
            "total_observations": self._total_observations,
            "unique_patterns": len(self._patterns),
        }
