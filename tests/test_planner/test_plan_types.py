"""Tests for app.planner.plan_types — model instantiation and enum sanity."""

from __future__ import annotations

from app.planner.plan_types import (
    GeneratePlan,
    LoadStrategy,
    MemoryState,
    Modality,
    ModelProfile,
    ModelRole,
    RoutingDecision,
    SessionState,
    SpeculativeMode,
    SpeedDecision,
    SpeedPriority,
)


# ---------------------------------------------------------------------------
# ModelProfile
# ---------------------------------------------------------------------------

def test_model_profile_creation(sample_model_profile: ModelProfile) -> None:
    """ModelProfile can be created with realistic data and fields are correct."""
    mp = sample_model_profile
    assert mp.name == "qwen_main"
    assert mp.path == "mlx-community/Qwen3.5-9B-MLX-4bit"
    assert mp.modality is Modality.TEXT
    assert mp.role is ModelRole.MAIN
    assert mp.load_strategy is LoadStrategy.HOT
    assert mp.supports_vision is False
    assert mp.estimated_model_ram_gb == 5.6
    assert mp.max_context_tokens_hard == 65536
    assert mp.max_output_tokens_hard == 16384


# ---------------------------------------------------------------------------
# SessionState
# ---------------------------------------------------------------------------

def test_session_state_defaults() -> None:
    """SessionState with only the required field uses sensible defaults."""
    ss = SessionState(session_id="minimal")
    assert ss.session_id == "minimal"
    assert ss.model_last_used == ""
    assert ss.raw_messages == []
    assert ss.compressed_summary is None
    assert ss.important_facts == []
    assert ss.prefix_cache_key is None
    assert ss.ctrsp_state_key is None
    assert ss.total_token_budget_used == 0
    assert ss.last_used_at == 0.0


# ---------------------------------------------------------------------------
# GeneratePlan
# ---------------------------------------------------------------------------

def test_generate_plan_creation(sample_generate_plan: GeneratePlan) -> None:
    """GeneratePlan can be created and routing / speed sub-models are accessible."""
    gp = sample_generate_plan
    assert gp.routing.selected_model == "qwen_main"
    assert gp.routing.use_sub_model is False
    assert gp.speed.speed_priority is SpeedPriority.NORMAL
    assert gp.speed.speculative_mode is SpeculativeMode.AUTO
    assert gp.effective_max_context_tokens == 65536
    assert gp.effective_max_output_tokens == 4096
    assert gp.enable_prefix_cache is True
    assert gp.should_reject is False
    assert gp.reject_reason is None


# ---------------------------------------------------------------------------
# Enum sanity
# ---------------------------------------------------------------------------

def test_memory_state_enum() -> None:
    """MemoryState enum has the expected members and string values."""
    assert MemoryState.GREEN.value == "green"
    assert MemoryState.YELLOW.value == "yellow"
    assert MemoryState.RED.value == "red"
    assert len(MemoryState) == 3


def test_speed_priority_enum() -> None:
    """SpeedPriority enum has the expected members and string values."""
    assert SpeedPriority.NORMAL.value == "normal"
    assert SpeedPriority.HIGH.value == "high"
    assert SpeedPriority.MAX_LONG_GENERATION.value == "max_long_generation"
    assert SpeedPriority.VISION_FAST_PATH.value == "vision_fast_path"
    assert len(SpeedPriority) == 4
