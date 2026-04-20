"""Long-Generation Optimizer (SPEC Pillar E).

Optimizes settings for long output generation:
- Suppresses context compaction during generation
- Preserves cache entries
- Prioritizes sustained tok/s over TTFT
"""

from __future__ import annotations
from dataclasses import dataclass
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

@dataclass
class LongGenConfig:
    """Configuration for long generation mode."""
    suppress_compaction: bool = True
    preserve_cache: bool = True
    prefer_speculative: bool = True
    stream_chunk_size: int = 1  # tokens per SSE chunk
    max_kv_size: int | None = None  # None = no limit

class LongGenerationOptimizer:
    """Manages long generation optimization settings."""

    def __init__(self):
        self._active = False
        self._config = LongGenConfig()

    def should_activate(self, max_output_tokens: int) -> bool:
        """Determine if long-gen mode should be activated."""
        settings = get_settings()
        threshold = settings.generation.default_max_output_tokens  # 4096
        return max_output_tokens >= threshold

    def activate(self, max_output_tokens: int) -> LongGenConfig:
        """Activate long generation mode with appropriate settings."""
        self._active = True
        self._config = LongGenConfig(
            suppress_compaction=True,
            preserve_cache=True,
            prefer_speculative=max_output_tokens >= 4096,
            stream_chunk_size=1,
        )
        logger.info(
            "Long-gen optimizer activated: max_output=%d, speculative=%s",
            max_output_tokens, self._config.prefer_speculative,
        )
        return self._config

    def deactivate(self) -> None:
        """Deactivate long generation mode."""
        self._active = False
        self._config = LongGenConfig()

    def is_active(self) -> bool:
        return self._active

    def get_config(self) -> LongGenConfig:
        return self._config

    def get_generation_kwargs(self) -> dict:
        """Return kwargs to pass to the generation backend."""
        if not self._active:
            return {}
        kwargs = {}
        if self._config.max_kv_size is not None:
            kwargs["max_kv_size"] = self._config.max_kv_size
        return kwargs
