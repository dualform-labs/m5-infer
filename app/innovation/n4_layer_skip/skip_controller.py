"""N4: Adaptive Layer Skipping — Runtime skip controller.

During decode, decides per-token whether to skip low-impact layers.
Easy tokens (high confidence) skip more layers; hard tokens use all layers.
"""

from __future__ import annotations
from app.innovation.n4_layer_skip.profiler import LayerProfiler
from app.core.logging import get_logger

logger = get_logger(__name__)

class SkipController:
    """Controls layer skipping during inference.

    Maintains a rolling "is_easy_token" signal from recent logit confidence.
    Callers set_last_confidence(prob) after each sampled token; should_skip_layer
    then consults that signal instead of the hardcoded True from before.
    """

    # Confidence threshold above which a token is considered "easy"
    # (used a separate knob from N4 layer-similarity threshold).
    EASY_PROB_THRESHOLD = 0.85

    def __init__(self, profiler: LayerProfiler):
        self._profiler = profiler
        self._active = False
        self._last_confidence: float = 1.0  # first token: be optimistic
        self._stats = {
            "tokens_processed": 0,
            "layers_skipped": 0,
            "total_layers": 0,
            "hard_tokens": 0,
            "easy_tokens": 0,
        }

    def activate(self) -> None:
        if self._profiler.is_profiled():
            self._active = True
            logger.info(
                "N4: Layer skipping activated (%d layers skippable, easy_prob>=%.2f)",
                self._profiler.skip_count(), self.EASY_PROB_THRESHOLD,
            )

    def deactivate(self) -> None:
        self._active = False

    def set_last_confidence(self, top1_prob: float) -> None:
        """Caller updates this with softmax top-1 prob of the just-sampled token.
        The NEXT token's layer-skip decision uses this signal.
        """
        self._last_confidence = float(top1_prob)
        if top1_prob >= self.EASY_PROB_THRESHOLD:
            self._stats["easy_tokens"] += 1
        else:
            self._stats["hard_tokens"] += 1

    def should_skip_layer(self, layer_idx: int, is_easy_token: bool | None = None) -> bool:
        """Determine if this layer should be skipped for the current token.

        If caller passes is_easy_token=None (recommended), the decision uses the
        rolling confidence set via set_last_confidence. The previous hard-coded
        True meant EVERY token skipped, losing the precision guard.
        """
        if not self._active:
            return False
        if not self._profiler.is_profiled():
            return False

        mask = self._profiler.get_skip_mask()
        if layer_idx >= len(mask):
            return False

        # Resolve is_easy_token from rolling confidence if caller didn't specify
        if is_easy_token is None:
            is_easy_token = self._last_confidence >= self.EASY_PROB_THRESHOLD
        if not is_easy_token:
            return False

        skip = mask[layer_idx]
        self._stats["total_layers"] += 1
        if skip:
            self._stats["layers_skipped"] += 1
        return skip

    def report_token(self) -> None:
        self._stats["tokens_processed"] += 1

    def get_skip_rate(self) -> float:
        if self._stats["total_layers"] == 0:
            return 0.0
        return self._stats["layers_skipped"] / self._stats["total_layers"]

    def get_stats(self) -> dict:
        return {**self._stats, "skip_rate": self.get_skip_rate(), "active": self._active}
