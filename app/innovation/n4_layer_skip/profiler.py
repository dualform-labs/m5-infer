"""N4: Adaptive Layer Skipping — Layer impact profiler.

Measures the residual similarity of each layer to identify
low-impact layers that can be skipped during decode.
"""

from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class LayerProfiler:
    """Profiles layer impact by measuring input-output similarity."""

    def __init__(self, threshold: float | None = None):
        if threshold is None:
            settings = get_settings()
            threshold = settings.innovation.n4_als_similarity_threshold
        self._threshold = threshold
        self._skip_mask: list[bool] = []  # True = skip this layer
        self._similarities: list[float] = []
        self._profiled = False

    def profile(self, model: nn.Module, sample_input: mx.array) -> list[bool]:
        """Profile all layers with a sample input to build skip mask.

        Runs a forward pass capturing input/output of each layer,
        then computes cosine similarity to find low-impact layers.
        """
        similarities = []
        skip_mask = []

        # Get the model's layers (typically model.model.layers)
        layers = None
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'layers'):
            layers = model.layers

        if layers is None:
            logger.warning("N4: Could not find model layers for profiling")
            self._profiled = False
            return []

        # Forward pass through each layer, measuring similarity
        hidden = sample_input
        for i, layer in enumerate(layers):
            try:
                layer_input = hidden
                # Run layer forward
                hidden = layer(hidden)
                if isinstance(hidden, tuple):
                    hidden = hidden[0]

                # Compute cosine similarity between input and output
                sim = _cosine_similarity(layer_input, hidden)
                similarities.append(sim)

                # High similarity = layer barely changes the hidden state = skippable
                should_skip = sim >= self._threshold
                skip_mask.append(should_skip)

            except Exception as e:
                logger.warning("N4: Error profiling layer %d: %s", i, e)
                similarities.append(0.0)
                skip_mask.append(False)

        self._similarities = similarities
        self._skip_mask = skip_mask
        self._profiled = True

        skip_count = sum(skip_mask)
        logger.info(
            "N4: Profiled %d layers, %d skippable (%.1f%%), threshold=%.4f",
            len(layers), skip_count, skip_count / len(layers) * 100, self._threshold,
        )

        return skip_mask

    def get_skip_mask(self) -> list[bool]:
        return self._skip_mask

    def get_similarities(self) -> list[float]:
        return self._similarities

    def is_profiled(self) -> bool:
        return self._profiled

    def skip_count(self) -> int:
        return sum(self._skip_mask)

    def get_stats(self) -> dict:
        return {
            "profiled": self._profiled,
            "total_layers": len(self._skip_mask),
            "skippable": self.skip_count(),
            "skip_rate": self.skip_count() / len(self._skip_mask) if self._skip_mask else 0,
            "threshold": self._threshold,
        }


def _cosine_similarity(a: mx.array, b: mx.array) -> float:
    """Compute cosine similarity between two hidden state tensors."""
    # Flatten to 1D for similarity computation
    a_flat = a.reshape(-1).astype(mx.float32)
    b_flat = b.reshape(-1).astype(mx.float32)

    dot = mx.sum(a_flat * b_flat)
    norm_a = mx.sqrt(mx.sum(a_flat * a_flat))
    norm_b = mx.sqrt(mx.sum(b_flat * b_flat))

    sim = dot / (norm_a * norm_b + 1e-8)
    # Materialize the lazy MLX computation
    mx.eval(sim)
    return sim.item()
