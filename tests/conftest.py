"""Shared pytest fixtures for the M5 MLX Inference Engine test suite."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

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
# tmp_config_dir — temporary directory with minimal valid config files
# ---------------------------------------------------------------------------

_MINIMAL_ENGINE_TOML = textwrap.dedent("""\
    [server]
    host = "127.0.0.1"
    port = 11435

    [engine]
    max_concurrent_requests = 1
    request_timeout_sec = 60
    stall_timeout_sec = 10
    session_idle_ttl_sec = 60

    [memory]
    total_budget_gb = 24
    green_threshold_gb = 18.0
    yellow_threshold_gb = 20.0
    red_threshold_gb = 22.0
    reserve_for_system_gb = 2.0
    allow_sub_model_only_if_free_gb = 6.0

    [context]
    summary_trigger_tokens = 24000
    raw_recent_turns = 8
    compress_tool_outputs = true

    [generation]
    default_max_output_tokens = 4096
    max_output_tokens_hard = 16384
    preserve_requested_max_tokens = true

    [cache]
    enable_prefix_cache = true
    enable_session_kv_cache = true
    prefix_cache_max_entries = 32
    session_cache_max_entries = 8
    preserve_active_kv = true

    [speculative]
    enabled = false
    mode = "auto"
    auto_enable_for_long_generation = true
    disable_when_memory_pressure = true
    draft_model_path = ""

    [planner]
    route_image_input_to_sub = true
    route_heavy_reasoning_to_sub = true
    fallback_to_main_if_sub_fails = true
    compact_before_sub_load = true
    prefer_cache_preservation = true

    [innovation]
    n1_ctrsp_enabled = false
    x4_context_fold_enabled = false
    x2_dpc_enabled = false
    n4_als_enabled = false
    n3_ssee_enabled = false
    x5r_compiled_enabled = false
    n6_pes_enabled = false
    n5_erp_enabled = false
    x1r_chunked_prefill_enabled = false
    x3_kv_distill_enabled = false
    n2_ggsa_enabled = false
""")

_MINIMAL_MODELS_TOML = textwrap.dedent("""\
    [models.main]
    name = "qwen_main"
    path = "mlx-community/Qwen3.5-9B-MLX-4bit"
    modality = "text"
    role = "main"
    load_strategy = "hot"
    supports_vision = false
    estimated_model_ram_gb = 5.6
    max_context_tokens_hard = 65536
    max_output_tokens_hard = 16384

    [models.sub_heavy]
    name = "gemma_heavy"
    path = "dealignai/Gemma-4-26B-A4B-JANG_4M-CRACK"
    modality = "image_text"
    role = "sub_heavy"
    load_strategy = "cold"
    supports_vision = true
    estimated_model_ram_gb = 15.5
    max_context_tokens_hard = 32768
    max_output_tokens_hard = 8192

    [sub_model]
    mode = "ephemeral"
    warm_ttl_sec = 180
""")


@pytest.fixture()
def tmp_config_dir(tmp_path: Path) -> Path:
    """Create a temporary directory containing minimal valid engine.toml and models.toml."""
    engine_path = tmp_path / "engine.toml"
    models_path = tmp_path / "models.toml"
    engine_path.write_text(_MINIMAL_ENGINE_TOML, encoding="utf-8")
    models_path.write_text(_MINIMAL_MODELS_TOML, encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# mock_settings — patches app.core.config.get_settings
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_settings(tmp_config_dir: Path) -> dict[str, Any]:
    """Patch ``app.core.config.get_settings`` to return a lightweight test config.

    Returns the settings dict so tests can inspect / override values.
    """
    settings: dict[str, Any] = {
        "config_dir": str(tmp_config_dir),
        "server": {"host": "127.0.0.1", "port": 11435},
        "engine": {
            "max_concurrent_requests": 1,
            "request_timeout_sec": 60,
            "stall_timeout_sec": 10,
            "session_idle_ttl_sec": 60,
        },
        "memory": {
            "total_budget_gb": 24,
            "green_threshold_gb": 18.0,
            "yellow_threshold_gb": 20.0,
            "red_threshold_gb": 22.0,
            "reserve_for_system_gb": 2.0,
            "allow_sub_model_only_if_free_gb": 6.0,
        },
        "generation": {
            "default_max_output_tokens": 4096,
            "max_output_tokens_hard": 16384,
        },
        "cache": {
            "enable_prefix_cache": True,
            "enable_session_kv_cache": True,
        },
        "speculative": {
            "enabled": False,
            "mode": "auto",
        },
    }
    with patch("app.core.config.get_settings", return_value=settings):
        yield settings


# ---------------------------------------------------------------------------
# sample_model_profile — realistic ModelProfile for the qwen main model
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_model_profile() -> ModelProfile:
    """Return a ``ModelProfile`` mirroring the Qwen main model from models.toml."""
    return ModelProfile(
        name="qwen_main",
        path="mlx-community/Qwen3.5-9B-MLX-4bit",
        modality=Modality.TEXT,
        role=ModelRole.MAIN,
        load_strategy=LoadStrategy.HOT,
        supports_vision=False,
        estimated_model_ram_gb=5.6,
        max_context_tokens_hard=65536,
        max_output_tokens_hard=16384,
    )


# ---------------------------------------------------------------------------
# sample_session_state — realistic SessionState
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_session_state() -> SessionState:
    """Return a ``SessionState`` with typical initial values."""
    return SessionState(
        session_id="test-session-001",
        model_last_used="qwen_main",
        raw_messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        compressed_summary=None,
        important_facts=["User prefers concise responses"],
        total_token_budget_used=128,
    )


# ---------------------------------------------------------------------------
# sample_generate_plan — realistic GeneratePlan
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_generate_plan() -> GeneratePlan:
    """Return a ``GeneratePlan`` for the qwen main model in a normal run."""
    return GeneratePlan(
        routing=RoutingDecision(
            selected_model="qwen_main",
            use_sub_model=False,
            requires_sub_load=False,
            reason="default main-model routing",
        ),
        speed=SpeedDecision(
            speed_priority=SpeedPriority.NORMAL,
            preserve_cache_priority=True,
            use_speculative=False,
            speculative_mode=SpeculativeMode.AUTO,
        ),
        effective_max_context_tokens=65536,
        effective_max_output_tokens=4096,
        should_compact_before_run=False,
        enable_prefix_cache=True,
        enable_session_kv=True,
        enable_ctrsp=False,
        enable_context_fold=False,
        crash_safe_margin_enabled=True,
        should_reject=False,
        reject_reason=None,
    )
