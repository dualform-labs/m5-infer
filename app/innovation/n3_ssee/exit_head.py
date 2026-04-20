"""N3: Self-Speculative Early Exit — Early exit logits head.

Uses the model's first 4 layers (3 GatedDeltaNet + 1 Full Attention)
as a fast draft predictor. The output at layer 3 is projected to logits
via a lightweight exit head.
"""

from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn
from app.core.logging import get_logger

logger = get_logger(__name__)

class EarlyExitHead:
    """Produces logits from early layer outputs for draft prediction."""

    def __init__(self, exit_layer_idx: int = 3):
        self._exit_layer_idx = exit_layer_idx
        self._head = None  # nn.Linear or similar
        self._norm = None  # Final RMSNorm/LayerNorm before LM head
        self._ready = False

    def initialize_from_model(self, model: nn.Module) -> None:
        """Initialize exit head using the model's existing final norm + LM head.

        Most decoder architectures apply a final norm (RMSNorm) before the
        LM head. We must apply the same norm to early-exit hidden states
        to produce calibrated logits.
        """
        # Find the LM head
        lm_head = None
        final_norm = None

        if hasattr(model, 'lm_head'):
            lm_head = model.lm_head
        elif hasattr(model, 'model') and hasattr(model.model, 'lm_head'):
            lm_head = model.model.lm_head

        # Find the final norm (typically model.model.norm or model.norm)
        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            final_norm = model.model.norm
        elif hasattr(model, 'norm'):
            final_norm = model.norm

        if lm_head is not None:
            self._head = lm_head
            self._norm = final_norm
            self._ready = True
            has_norm = "with" if final_norm else "WITHOUT"
            logger.info(
                "N3: Early exit head initialized %s final norm (layer %d)",
                has_norm, self._exit_layer_idx,
            )
        else:
            logger.warning("N3: Could not find model LM head")

    def predict(self, hidden_state: mx.array) -> mx.array:
        """Produce logits from an early layer's hidden state.

        Applies final norm (if found) before the LM head projection,
        matching the model's normal output pipeline.
        """
        if not self._ready or self._head is None:
            raise RuntimeError("Exit head not initialized")
        x = hidden_state
        if self._norm is not None:
            x = self._norm(x)
        return self._head(x)

    @property
    def exit_layer_idx(self) -> int:
        return self._exit_layer_idx

    def is_ready(self) -> bool:
        return self._ready
