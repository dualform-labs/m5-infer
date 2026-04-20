"""Speed strategy selection for the M5 MLX Inference Engine (SPEC section 8.2).

Determines which latency / throughput knobs to enable for a single
generation pass based on request parameters, memory state, and engine
configuration.
"""

from __future__ import annotations

from app.core.config import get_settings
from app.core.logging import get_logger
from app.planner.plan_types import (
    MemorySnapshot,
    MemoryState,
    SpeedDecision,
    SpeedPriority,
    SpeculativeMode,
)

logger = get_logger(__name__)


def select_speed_strategy(
    prompt_token_count: int = 0,
    max_output_tokens: int = 4096,
    has_image: bool = False,
    session_is_continuation: bool = False,
    memory_state: MemorySnapshot | None = None,
    speed_priority_hint: str | None = None,
    prefer_long_generation: bool = False,
    disable_speculative: bool = False,
    force_cache_preserve: bool = False,
) -> SpeedDecision:
    """Select the speed optimization strategy for a request.

    Parameters
    ----------
    prompt_token_count:
        Number of prompt tokens (used for priority heuristics).
    max_output_tokens:
        Requested maximum output tokens for the generation.
    has_image:
        Whether the request includes an image input.
    session_is_continuation:
        True when this turn continues an existing conversation.
    memory_state:
        Current unified-memory snapshot (may be ``None`` if unknown).
    speed_priority_hint:
        Explicit priority string from the caller (maps to
        :class:`SpeedPriority`).
    prefer_long_generation:
        Hint that the caller expects a long output.
    disable_speculative:
        Force speculative decoding off regardless of config.
    force_cache_preserve:
        Force cache-preservation priority on.

    Returns
    -------
    SpeedDecision
        Fully resolved speed / throughput knobs for the generation pass.
    """
    settings = get_settings()
    innovation = settings.innovation

    # ------------------------------------------------------------------
    # 1. Determine base priority
    # ------------------------------------------------------------------
    if speed_priority_hint:
        try:
            priority = SpeedPriority(speed_priority_hint)
        except ValueError:
            logger.warning(
                "Unknown speed_priority_hint %r, falling back to NORMAL",
                speed_priority_hint,
            )
            priority = SpeedPriority.NORMAL
    elif has_image:
        priority = SpeedPriority.VISION_FAST_PATH
    elif prefer_long_generation or max_output_tokens >= 4096:
        priority = SpeedPriority.MAX_LONG_GENERATION
    elif session_is_continuation or prompt_token_count >= 4096:
        priority = SpeedPriority.HIGH
    else:
        priority = SpeedPriority.NORMAL

    # ------------------------------------------------------------------
    # 2. Determine speculative decoding mode
    # ------------------------------------------------------------------
    is_memory_pressure = (
        memory_state is not None
        and memory_state.state != MemoryState.GREEN
    )

    if disable_speculative or not settings.speculative.enabled:
        spec_mode = SpeculativeMode.DISABLED
    elif is_memory_pressure and settings.speculative.disable_when_memory_pressure:
        spec_mode = SpeculativeMode.DISABLED
    elif settings.speculative.mode == "auto":
        # Auto: prefer SSEE (no extra memory cost), then draft model
        if innovation.n3_ssee_enabled:
            spec_mode = SpeculativeMode.SSEE
        elif settings.speculative.draft_model_path:
            spec_mode = SpeculativeMode.DRAFT_MODEL
        else:
            spec_mode = SpeculativeMode.DISABLED
    else:
        try:
            spec_mode = SpeculativeMode(settings.speculative.mode)
        except ValueError:
            spec_mode = SpeculativeMode.DISABLED

    use_speculative = spec_mode != SpeculativeMode.DISABLED

    # ------------------------------------------------------------------
    # 3. Dynamic Prompt Caching (DPC / X2)
    # ------------------------------------------------------------------
    use_dpc = (
        innovation.x2_dpc_enabled
        and not (
            is_memory_pressure
            and memory_state is not None
            and memory_state.state == MemoryState.RED
        )
    )

    # ------------------------------------------------------------------
    # 4. Adaptive Layer Skipping (N4)
    # ------------------------------------------------------------------
    use_layer_skip = innovation.n4_als_enabled

    # ------------------------------------------------------------------
    # 5. Cache preservation
    # ------------------------------------------------------------------
    preserve_cache = (
        force_cache_preserve
        or priority in (SpeedPriority.HIGH, SpeedPriority.MAX_LONG_GENERATION)
        or settings.planner.prefer_cache_preservation
    )

    # ------------------------------------------------------------------
    # 6. Long generation optimizer
    # ------------------------------------------------------------------
    prefer_long_gen = priority == SpeedPriority.MAX_LONG_GENERATION

    logger.debug(
        "speed_strategy selected: priority=%s spec=%s dpc=%s layer_skip=%s "
        "cache_preserve=%s long_gen=%s",
        priority.value,
        spec_mode.value,
        use_dpc,
        use_layer_skip,
        preserve_cache,
        prefer_long_gen,
    )

    return SpeedDecision(
        speed_priority=priority,
        preserve_cache_priority=preserve_cache,
        use_speculative=use_speculative,
        speculative_mode=spec_mode,
        use_dpc=use_dpc,
        use_layer_skip=use_layer_skip,
        prefer_long_generation_optimizer=prefer_long_gen,
    )
