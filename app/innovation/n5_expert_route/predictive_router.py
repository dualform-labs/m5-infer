"""N5: Expert Route Prediction — Predictive expert pre-loading."""

from __future__ import annotations
from app.innovation.n5_expert_route.route_profiler import RouteProfiler
from app.core.logging import get_logger

logger = get_logger(__name__)

class PredictiveRouter:
    """Predicts and pre-fetches expert weights based on routing patterns."""

    def __init__(self, profiler: RouteProfiler, min_confidence: float = 0.6):
        self._profiler = profiler
        self._min_confidence = min_confidence
        self._stats = {"predictions": 0, "hits": 0, "misses": 0, "skipped": 0}

    def predict_experts(self, token_id: int, prev_token_id: int | None) -> list[int] | None:
        """Predict which experts will be needed for the next token.

        Returns predicted expert indices, or None if prediction is low-confidence.
        """
        confidence = self._profiler.get_prediction_confidence(token_id, prev_token_id)
        if confidence < self._min_confidence:
            self._stats["skipped"] += 1
            return None

        prediction = self._profiler.predict(token_id, prev_token_id)
        if prediction:
            self._stats["predictions"] += 1
        return prediction

    def record_actual(self, predicted: list[int] | None, actual: list[int]) -> None:
        """Record whether prediction was correct."""
        if predicted is None:
            return
        if set(predicted) == set(actual):
            self._stats["hits"] += 1
        else:
            self._stats["misses"] += 1

    def get_accuracy(self) -> float:
        total = self._stats["hits"] + self._stats["misses"]
        if total == 0:
            return 0.0
        return self._stats["hits"] / total

    def get_stats(self) -> dict:
        return {**self._stats, "accuracy": self.get_accuracy()}
