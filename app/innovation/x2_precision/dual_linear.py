"""X2 DPC — DualPrecisionLinear: runtime-switchable 4-bit / 2-bit forward.

A drop-in replacement for `mlx.nn.QuantizedLinear` that holds both the
original 4-bit weights AND the pre-computed 2-bit shadow for the same layer.
At forward time it picks one based on a shared `DualPrecisionPolicy` flag.

Usage:
    policy = DualPrecisionPolicy()
    shadow = ShadowWeights(); shadow.generate(model)
    install_dual_precision(model, shadow, policy,
                           name_filter=lambda n: n.endswith(('up_proj','down_proj','gate_proj')))

    # Before each generation step (typically in decode loop):
    policy.use_2bit = confidence_router.should_use_2bit(prev_logits)

    # model(tokens, cache=cache) will route FFN layers to 2-bit when the flag is set.

Safety:
  - Only activated when `name_filter` returns True (default: FFN only).
  - Attention layers are intentionally excluded (sensitive to quantization).
  - Policy defaults to 4-bit (off) so existing callers see no change.
"""

from __future__ import annotations

from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn

from app.core.logging import get_logger
from app.innovation.x2_precision.shadow_weights import ShadowWeights

logger = get_logger(__name__)


class DualPrecisionPolicy:
    """Tiny shared flag consulted by every DualPrecisionLinear forward."""

    def __init__(self) -> None:
        self.use_2bit: bool = False
        self.stats = {"q4_calls": 0, "q2_calls": 0}

    def reset_stats(self) -> None:
        self.stats = {"q4_calls": 0, "q2_calls": 0}


class DualPrecisionLinear(nn.Module):
    """Holds both 4-bit primary and 2-bit shadow weights for a single Linear.

    This does NOT inherit QuantizedLinear to avoid mlx Module registration
    complications; it re-implements forward via `mx.quantized_matmul`.
    """

    def __init__(
        self,
        q4_weight: mx.array, q4_scales: mx.array, q4_biases: mx.array,
        q2_weight: mx.array, q2_scales: mx.array, q2_biases: mx.array,
        group_size: int,
        policy: DualPrecisionPolicy,
        q4_bias: mx.array | None = None,
    ) -> None:
        super().__init__()
        # 4-bit primary
        self.q4_weight = q4_weight
        self.q4_scales = q4_scales
        self.q4_biases = q4_biases
        # 2-bit shadow
        self.q2_weight = q2_weight
        self.q2_scales = q2_scales
        self.q2_biases = q2_biases
        self.group_size = group_size
        self.q4_bias = q4_bias  # optional linear bias (separate from quant biases)
        # Policy is a module attribute but NOT an mlx parameter — skipped by freeze
        self._policy = policy

    @classmethod
    def from_quantized(
        cls,
        q_linear: nn.Module,
        shadow_triplet: tuple[mx.array, mx.array, mx.array],
        policy: DualPrecisionPolicy,
    ) -> "DualPrecisionLinear":
        q2_w, q2_s, q2_b = shadow_triplet
        inst = cls(
            q4_weight=q_linear.weight,
            q4_scales=q_linear.scales,
            q4_biases=q_linear.biases,
            q2_weight=q2_w, q2_scales=q2_s, q2_biases=q2_b,
            group_size=q_linear.group_size,
            policy=policy,
            q4_bias=getattr(q_linear, "bias", None),
        )
        return inst

    def __call__(self, x: mx.array) -> mx.array:
        if self._policy.use_2bit:
            self._policy.stats["q2_calls"] += 1
            y = mx.quantized_matmul(
                x, self.q2_weight, self.q2_scales, self.q2_biases,
                transpose=True, group_size=self.group_size, bits=2,
            )
        else:
            self._policy.stats["q4_calls"] += 1
            y = mx.quantized_matmul(
                x, self.q4_weight, self.q4_scales, self.q4_biases,
                transpose=True, group_size=self.group_size, bits=4,
            )
        if self.q4_bias is not None:
            y = y + self.q4_bias
        return y


def install_dual_precision(
    model: nn.Module,
    shadow: ShadowWeights,
    policy: DualPrecisionPolicy,
    name_filter: Optional[Callable[[str], bool]] = None,
) -> int:
    """Walk the model, replacing filter-matching QuantizedLinear with
    DualPrecisionLinear instances backed by the corresponding shadow.

    Returns the count of layers successfully swapped.
    """
    if name_filter is None:
        name_filter = lambda n: n.endswith(("up_proj", "down_proj", "gate_proj"))

    swapped = 0
    missing_shadow = 0
    # First pass: collect replacements (name, new_module).
    to_swap: list[tuple[str, DualPrecisionLinear]] = []
    for name, module in model.named_modules():
        if not name_filter(name):
            continue
        # Only swap true QuantizedLinear (not already dual).
        if not (hasattr(module, "weight") and hasattr(module, "scales") and
                hasattr(module, "biases") and getattr(module, "bits", None) == 4):
            continue
        shadow_triplet = shadow.get_shadow(name)
        if shadow_triplet is None:
            missing_shadow += 1
            continue
        dp = DualPrecisionLinear.from_quantized(module, shadow_triplet, policy)
        to_swap.append((name, dp))

    # Second pass: attach dp back into parent via setattr.
    for name, dp in to_swap:
        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            if p.isdigit():
                parent = parent[int(p)]
            else:
                parent = getattr(parent, p)
        setattr(parent, parts[-1], dp)
        swapped += 1

    logger.info(
        "X2 DPC: installed %d DualPrecisionLinear (missing_shadow=%d)",
        swapped, missing_shadow,
    )
    return swapped
