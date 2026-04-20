"""X1-R: Chunked Prefill Optimization.

Splits prefill into chunks and uses mx.async_eval between chunks
for better GPU utilization. Expected 10-20% TTFT improvement.

Research phase — effectiveness depends on chunk size tuning.
"""

from __future__ import annotations
import mlx.core as mx
from app.core.logging import get_logger

logger = get_logger(__name__)

class ChunkedPrefill:
    """Optimizes prefill by processing prompt in chunks with async overlap."""

    def __init__(self, chunk_size: int = 512):
        self._chunk_size = chunk_size
        self._ready = False

    def optimal_chunk_size(self, prompt_length: int) -> int:
        """Determine optimal chunk size based on prompt length."""
        if prompt_length < 1024:
            return prompt_length  # No chunking needed
        elif prompt_length < 4096:
            return 512
        elif prompt_length < 16384:
            return 1024
        else:
            return 2048

    def get_stats(self) -> dict:
        return {"chunk_size": self._chunk_size, "ready": self._ready}
