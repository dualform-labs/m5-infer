"""X3: Semantic KV Distillation — Mean-pooling prototype.

Compresses old KV cache entries by mean-pooling groups of entries
into single 'distilled' entries. This is the simple baseline before
attempting learned MLP distillation.

Research phase — quality impact needs measurement.
"""

from __future__ import annotations
import mlx.core as mx
from app.core.logging import get_logger

logger = get_logger(__name__)

class MeanPoolDistiller:
    """Distills KV cache entries via mean pooling."""

    def __init__(self, pool_size: int = 8):
        self._pool_size = pool_size
        self._stats = {"distilled_entries": 0, "original_entries": 0}

    def distill(self, kv_entries: mx.array, pool_size: int | None = None) -> mx.array:
        """Compress KV entries by mean-pooling groups.

        Args:
            kv_entries: [num_entries, head_dim] or [num_heads, num_entries, head_dim]
            pool_size: Number of entries to merge into one

        Returns:
            Compressed KV with ceil(num_entries / pool_size) entries
        """
        ps = pool_size or self._pool_size

        if kv_entries.ndim == 2:
            n, d = kv_entries.shape
            if n <= ps:
                return kv_entries
            # Pad to multiple of pool_size
            pad_n = (ps - n % ps) % ps
            if pad_n > 0:
                padding = mx.zeros((pad_n, d))
                kv_entries = mx.concatenate([kv_entries, padding], axis=0)
            # Reshape and mean
            reshaped = kv_entries.reshape(-1, ps, d)
            distilled = mx.mean(reshaped, axis=1)
            self._stats["original_entries"] += n
            self._stats["distilled_entries"] += distilled.shape[0]
            return distilled

        elif kv_entries.ndim == 3:
            h, n, d = kv_entries.shape
            if n <= ps:
                return kv_entries
            results = []
            for i in range(h):
                results.append(self.distill(kv_entries[i], ps))
            return mx.stack(results)

        return kv_entries

    def compression_ratio(self) -> float:
        if self._stats["original_entries"] == 0:
            return 1.0
        return self._stats["distilled_entries"] / self._stats["original_entries"]

    def get_stats(self) -> dict:
        return {**self._stats, "compression_ratio": self.compression_ratio(), "pool_size": self._pool_size}
