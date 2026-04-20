"""Data models for the M5 MLX Inference Engine planner (SPEC section 7)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ModelRole(str, Enum):
    """Role a model plays in the inference pipeline."""

    MAIN = "main"
    SUB_HEAVY = "sub_heavy"
    SUB_SPEED = "sub_speed"


class Modality(str, Enum):
    """Input modality the model supports."""

    TEXT = "text"
    IMAGE_TEXT = "image_text"
    DRAFT = "draft"


class LoadStrategy(str, Enum):
    """How the model should be loaded into memory."""

    HOT = "hot"
    COLD = "cold"


class MemoryState(str, Enum):
    """Traffic-light classification of unified memory pressure."""

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class SpeedPriority(str, Enum):
    """Requested speed / latency tier."""

    NORMAL = "normal"
    HIGH = "high"
    MAX_LONG_GENERATION = "max_long_generation"
    VISION_FAST_PATH = "vision_fast_path"


class SpeculativeMode(str, Enum):
    """Speculative decoding strategy."""

    AUTO = "auto"
    SSEE = "ssee"
    DRAFT_MODEL = "draft_model"
    DISABLED = "disabled"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ModelProfile(BaseModel):
    """Static profile describing a model available to the engine."""

    name: str
    path: str
    modality: Modality = Modality.TEXT
    role: ModelRole = ModelRole.MAIN
    load_strategy: LoadStrategy = LoadStrategy.HOT
    supports_vision: bool = False
    estimated_model_ram_gb: float = 0.0
    max_context_tokens_hard: int = 8192
    max_output_tokens_hard: int = 4096


class SessionState(BaseModel):
    """Per-session conversational state tracked across turns."""

    session_id: str
    model_last_used: str = ""
    raw_messages: list[dict] = Field(default_factory=list)
    compressed_summary: str | None = None
    important_facts: list[str] = Field(default_factory=list)
    prefix_cache_key: str | None = None
    ctrsp_state_key: str | None = None
    total_token_budget_used: int = 0
    last_used_at: float = 0.0


class MemorySnapshot(BaseModel):
    """Point-in-time snapshot of Apple Silicon unified memory."""

    total_used_gb: float = 0.0
    available_gb: float = 0.0
    worker_used_gb: float = 0.0
    cache_used_gb: float = 0.0
    state: MemoryState = MemoryState.GREEN


class RoutingDecision(BaseModel):
    """Which model to run and how to load it."""

    selected_model: str
    use_sub_model: bool = False
    requires_sub_load: bool = False
    reason: str = ""
    unload_after_run: bool = False
    use_vision_path: bool = False


class SpeedDecision(BaseModel):
    """Latency / throughput knobs for a single generation pass."""

    speed_priority: SpeedPriority = SpeedPriority.NORMAL
    preserve_cache_priority: bool = True
    use_speculative: bool = False
    speculative_mode: SpeculativeMode = SpeculativeMode.AUTO
    use_dpc: bool = False
    use_layer_skip: bool = False
    prefer_long_generation_optimizer: bool = False


class GeneratePlan(BaseModel):
    """Fully resolved plan consumed by the generation executor."""

    routing: RoutingDecision
    speed: SpeedDecision = Field(default_factory=SpeedDecision)
    effective_max_context_tokens: int = 8192
    effective_max_output_tokens: int = 4096
    should_compact_before_run: bool = False
    enable_prefix_cache: bool = True
    enable_session_kv: bool = True
    enable_ctrsp: bool = False
    enable_context_fold: bool = False
    crash_safe_margin_enabled: bool = True
    should_reject: bool = False
    reject_reason: str | None = None
    # Phase D FPQM — per-request KV precision override.
    # None = inherit engine default, "int4"/"bf16" explicitly selects.
    kv_precision_override: str | None = None
    # Phase D FPQM — when true, force full-precision path (bf16 KV, disable
    # speculative, disable DPC shadow swap) for lossless guarantee.
    prefer_quality: bool = False
