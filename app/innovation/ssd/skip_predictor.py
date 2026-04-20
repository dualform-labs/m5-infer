"""SSD Skip Predictor — runtime decision logic for token bypass."""

from __future__ import annotations

from app.core.logging import get_logger
from app.innovation.ssd.ngram_table import get_ssd_table

logger = get_logger(__name__)


class SkipPredictor:
    """Decides whether to skip the model for the next token.

    Strategy:
      - Look up recent context in SSD table
      - If high-confidence match exists, return predicted token
      - Periodically (every `verify_every` skips), still run model and
        compare — record correctness for monitoring

    Per-request instance (not singleton) so per-request stats are isolated.
    """

    def __init__(
        self,
        min_confidence: float = 0.95,
        verify_every: int = 4,
        enabled: bool = True,
    ):
        self._min_conf = min_confidence
        self._verify_every = max(1, verify_every)
        self._enabled = enabled
        self._table = get_ssd_table() if enabled else None
        self._n_skipped = 0
        self._n_verified = 0
        self._n_predicted_total = 0

    def predict(self, context_tokens: list[int]) -> int | None:
        """Return predicted token id if we should skip the model, else None.

        Caller pattern:
          predicted = skip_predictor.predict(generated_ids)
          if predicted is not None:
              # Skip model forward, emit `predicted` directly
              ...
          else:
              # Run model normally
              ...
        """
        if not self._enabled or self._table is None:
            return None
        # Skip if we're due for a verify check (every N-th skip)
        if self._n_skipped > 0 and self._n_skipped % self._verify_every == 0:
            # Force model to run — we'll verify it matches our prediction
            self._n_verified += 1
            return None
        predicted = self._table.lookup(context_tokens, self._min_conf)
        if predicted is not None:
            self._n_skipped += 1
            self._n_predicted_total += 1
        return predicted

    def record_verify_result(self, our_prediction: int, model_prediction: int) -> None:
        """Called during verify pass to record agreement."""
        if self._table is None:
            return
        was_correct = (our_prediction == model_prediction)
        self._table.record_verify(was_correct)
        if not was_correct:
            logger.debug(
                "SSD verify mismatch: predicted %d, model said %d",
                our_prediction, model_prediction,
            )

    def stats(self) -> dict:
        return {
            "enabled": self._enabled,
            "skipped": self._n_skipped,
            "verified": self._n_verified,
            "skip_rate": (self._n_predicted_total / max(1, self._n_skipped + self._n_verified)),
        }
