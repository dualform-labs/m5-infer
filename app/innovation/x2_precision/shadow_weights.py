"""X2: Dynamic Precision Cascading — 2-bit shadow weight generation.

At startup, pre-computes 2-bit quantized versions of all model layers.
These shadow weights are used for high-confidence tokens (top-1 prob > threshold),
giving 1.34x speedup per layer (M5 measured).
"""

from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn
from app.core.logging import get_logger

logger = get_logger(__name__)

class ShadowWeights:
    """Manages 2-bit shadow copies of model weights."""

    def __init__(self):
        self._shadows: dict[str, tuple[mx.array, mx.array, mx.array]] = {}
        self._ready = False
        self._memory_bytes = 0

    def generate(self, model: nn.Module) -> None:
        """Pre-compute 2-bit shadow weights for all QuantizedLinear layers."""
        count = 0
        total_bytes = 0

        for name, module in model.named_modules():
            if hasattr(module, 'weight') and hasattr(module, 'scales'):
                # This is a quantized layer - re-quantize to 2-bit
                # Get the dequantized weights first, then requantize
                try:
                    # For QuantizedLinear, we can access the quantized data
                    # and requantize at lower precision
                    if hasattr(module, 'bits') and module.bits == 4:
                        # Dequantize 4-bit -> float -> quantize 2-bit
                        w_full = mx.dequantize(
                            module.weight, module.scales, module.biases,
                            module.group_size, module.bits,
                        )
                        w_2bit, scales_2bit, biases_2bit = mx.quantize(w_full, bits=2)
                        self._shadows[name] = (w_2bit, scales_2bit, biases_2bit)
                        total_bytes += w_2bit.nbytes + scales_2bit.nbytes + biases_2bit.nbytes
                        count += 1
                except Exception as e:
                    logger.warning("Failed to create 2-bit shadow for %s: %s", name, e)
                    continue

        if count > 0:
            # Force materialization of all shadow weight tensors
            mx.eval(*[t for triple in self._shadows.values() for t in triple])
            self._ready = True
            self._memory_bytes = total_bytes
            logger.info(
                "DPC: Generated %d 2-bit shadow layers (%.2f MB)",
                count, total_bytes / 1e6,
            )
        else:
            logger.warning("DPC: No layers were converted to 2-bit shadows")

    def get_shadow(self, layer_name: str) -> tuple[mx.array, mx.array, mx.array] | None:
        """Get 2-bit shadow weights for a named layer."""
        return self._shadows.get(layer_name)

    def is_ready(self) -> bool:
        return self._ready

    def memory_bytes(self) -> int:
        return self._memory_bytes

    def clear(self) -> None:
        self._shadows.clear()
        self._ready = False
        self._memory_bytes = 0
        mx.clear_cache()
