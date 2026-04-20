from __future__ import annotations
from app.planner.plan_types import (
    GeneratePlan, RoutingDecision, SpeedDecision, SpeedPriority, SpeculativeMode, MemorySnapshot, MemoryState,
)
from app.planner.routing import route_request
from app.planner.speed_selector import select_speed_strategy
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class RequestPlanner:
    """Converts incoming requests into GeneratePlan instances."""

    def __init__(self, memory_guard=None):
        self._memory_guard = memory_guard

    def plan(
        self,
        messages: list[dict],
        max_tokens: int = 4096,
        prompt_token_count: int = 0,
        has_image: bool = False,
        force_heavy: bool = False,
        model_hint: str | None = None,
        speed_priority_hint: str | None = None,
        prefer_long_generation: bool = False,
        prefer_long_context: bool = False,
        disable_speculative: bool = False,
        force_cache_preserve: bool = False,
        session_is_continuation: bool = False,
        num_tool_schemas: int = 0,
        kv_precision: str | None = None,
        prefer_quality: bool = False,
    ) -> GeneratePlan:
        """Create a GeneratePlan for the given request parameters."""
        settings = get_settings()

        # 1. Route to model
        routing = route_request(
            has_image=has_image,
            force_heavy=force_heavy,
            model_hint=model_hint,
        )

        # 2. Get memory state if guard available
        memory_state = None
        if self._memory_guard:
            memory_state = self._memory_guard.check_memory()

        # 3. Select speed strategy
        speed = select_speed_strategy(
            prompt_token_count=prompt_token_count,
            max_output_tokens=max_tokens,
            has_image=has_image,
            session_is_continuation=session_is_continuation,
            memory_state=memory_state,
            speed_priority_hint=speed_priority_hint,
            prefer_long_generation=prefer_long_generation,
            disable_speculative=disable_speculative,
            force_cache_preserve=force_cache_preserve,
        )

        # 4. Determine effective token limits
        effective_max_context = settings.generation.max_output_tokens_hard
        effective_max_output = min(max_tokens, settings.generation.max_output_tokens_hard)

        # 5. Should we reject?
        should_reject = False
        reject_reason = None
        if memory_state and memory_state.state == MemoryState.RED:
            if routing.requires_sub_load:
                should_reject = True
                reject_reason = "Memory pressure too high for sub model load"

        # 6. Context optimization flags
        # CTRSP: enable if system prompt exists (benefits from state caching
        # on 2nd+ requests with same system prompt, even without explicit session_id)
        has_system_prompt = any(m.get("role") == "system" for m in messages)
        enable_ctrsp = (
            has_system_prompt
            and not routing.use_sub_model
            and settings.innovation.n1_ctrsp_enabled
        )
        enable_context_fold = (
            num_tool_schemas > 5
            and settings.innovation.x4_context_fold_enabled
        )

        # Phase D FPQM — resolve effective kv_precision for this request.
        # prefer_quality short-circuits to "bf16" regardless of kv_precision.
        effective_kv_precision = kv_precision
        if prefer_quality:
            effective_kv_precision = "bf16"

        plan = GeneratePlan(
            routing=routing,
            speed=speed,
            effective_max_context_tokens=effective_max_context,
            effective_max_output_tokens=effective_max_output,
            should_compact_before_run=routing.requires_sub_load and settings.planner.compact_before_sub_load,
            enable_prefix_cache=settings.cache.enable_prefix_cache,
            enable_session_kv=settings.cache.enable_session_kv_cache,
            enable_ctrsp=enable_ctrsp,
            enable_context_fold=enable_context_fold,
            crash_safe_margin_enabled=memory_state.state != MemoryState.GREEN if memory_state else False,
            should_reject=should_reject,
            reject_reason=reject_reason,
            kv_precision_override=effective_kv_precision,
            prefer_quality=prefer_quality,
        )

        logger.info(
            "Plan created: model=%s, speed=%s, ctrsp=%s, fold=%s, reject=%s",
            routing.selected_model,
            speed.speed_priority.value,
            enable_ctrsp,
            enable_context_fold,
            should_reject,
        )

        return plan
