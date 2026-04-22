"""M5 MLX Inference Engine — configuration loader.

Loads engine.toml and models.toml via :mod:`app.core.paths`, which handles
both source-tree (``./configs/``) and PyPI-installed (package-bundled
``app/_defaults/``) layouts. Exposes typed Pydantic settings via singleton
accessors.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, Literal

import tomlkit
from pydantic import BaseModel, Field

from app.core.paths import find_config


# ---------------------------------------------------------------------------
# Legacy convenience constants (best-effort; may be None on PyPI installs)
# ---------------------------------------------------------------------------

def _best_effort_project_root() -> Path | None:
    current = Path(__file__).resolve().parent
    for ancestor in (current, *current.parents):
        if (ancestor / "pyproject.toml").is_file():
            return ancestor
    return None


PROJECT_ROOT: Path | None = _best_effort_project_root()
CONFIGS_DIR: Path | None = (PROJECT_ROOT / "configs") if PROJECT_ROOT else None


# ---------------------------------------------------------------------------
# TOML helpers
# ---------------------------------------------------------------------------


def _load_toml(name: str) -> dict:
    """Load a config file via the layered search, falling back to bundled defaults."""
    path, text = find_config(name)
    if text is None:
        assert path is not None
        text = path.read_text(encoding="utf-8")
    return tomlkit.loads(text)


# ---------------------------------------------------------------------------
# Engine section models
# ---------------------------------------------------------------------------

class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 11436  # default port


class EngineConfig(BaseModel):
    max_concurrent_requests: int = 1
    request_timeout_sec: int = 600
    stall_timeout_sec: int = 20
    session_idle_ttl_sec: int = 300
    # v1.0 auto-tune: detect Apple Silicon tier at startup and override
    # chunk/concurrency/cache sizing to match the chip. Users can set
    # this to false to keep engine.toml values verbatim.
    auto_tune: bool = True


class MemoryConfig(BaseModel):
    total_budget_gb: int = 24
    green_threshold_gb: float = 18.0
    yellow_threshold_gb: float = 20.0
    red_threshold_gb: float = 22.0
    reserve_for_system_gb: float = 2.0
    allow_sub_model_only_if_free_gb: float = 6.0


class ContextConfig(BaseModel):
    summary_trigger_tokens: int = 24000
    raw_recent_turns: int = 8
    compress_tool_outputs: bool = True


class GenerationConfig(BaseModel):
    default_max_output_tokens: int = 4096
    max_output_tokens_hard: int = 16384
    preserve_requested_max_tokens: bool = True
    kv_bits: int | None = 4  # 4/8/None. Silently disabled when CTRSP is active.
    lookahead_k: int = 4  # N-gram draft size for speculative lookahead. 0 disables.
    # Thinking-aware budget: tokens inside <think>...</think> (Qwen3.5)
    # use this budget instead of max_tokens, so answers never run out of
    # tokens because reasoning was long. When the cap is hit we force-inject
    # </think> rather than terminating, so the answer still lands. 1536 is
    # the sweet spot for 9B-4bit — keeps decode fast (KV cache doesn't grow
    # past the memory-bandwidth cliff) while giving enough reasoning for
    # math, logic, and code tasks observed in internal quality evaluation.
    max_thinking_tokens: int = 1536


class CacheConfig(BaseModel):
    enable_prefix_cache: bool = True
    enable_session_kv_cache: bool = True
    prefix_cache_max_entries: int = 32
    session_cache_max_entries: int = 8
    preserve_active_kv: bool = True


class SpeculativeConfig(BaseModel):
    enabled: bool = False
    mode: Literal["auto", "always", "never"] = "auto"
    auto_enable_for_long_generation: bool = True
    disable_when_memory_pressure: bool = True
    draft_model_path: str = ""


class PlannerConfig(BaseModel):
    route_image_input_to_sub: bool = True
    route_heavy_reasoning_to_sub: bool = True
    fallback_to_main_if_sub_fails: bool = True
    compact_before_sub_load: bool = True
    prefer_cache_preservation: bool = True


class InnovationConfig(BaseModel):
    # N1 — Continuous Token-Release Scheduling with Preemption
    n1_ctrsp_enabled: bool = True
    # X4 — Context fold
    x4_context_fold_enabled: bool = True
    # X2 — Dynamic Prompt Caching
    x2_dpc_enabled: bool = True
    x2_dpc_confidence_threshold: float = 0.95
    # N4 — Adaptive Layer Skipping
    n4_als_enabled: bool = True
    n4_als_similarity_threshold: float = 0.99
    # N3 — Self-Speculative Early Exit
    n3_ssee_enabled: bool = True
    n3_ssee_min_acceptance_rate: float = 0.30
    n3_ssee_num_draft_tokens: int = 4
    # RDMS speculative draft size. Auto-tune scales this per bandwidth tier
    # (larger BW → more draft tokens profit per main forward pass).
    rdms_num_draft: int = 4
    # X5R — Compiled graph execution
    x5r_compiled_enabled: bool = True
    # N6 — Parallel Expert Scheduling
    n6_pes_enabled: bool = True
    n6_pes_num_parallel: int = 2
    # N5 — Entropy-Routed Pruning
    n5_erp_enabled: bool = True
    # X1R — Chunked prefill (off by default)
    x1r_chunked_prefill_enabled: bool = False
    # X3 — KV distillation (off by default)
    x3_kv_distill_enabled: bool = False
    # N2 — Grouped Greedy Speculative Attention (off by default)
    n2_ggsa_enabled: bool = False


# ---------------------------------------------------------------------------
# Tiered Memory Architecture — Runtime Mode
# ---------------------------------------------------------------------------

RuntimeMode = Literal["moderate", "aggressive", "extreme"]


class ModeFeatures(BaseModel):
    """Per-mode feature toggles."""
    ctrsp_lru_size: int = 32
    draft_model_path: str = ""              # 空 = RDMS 無効
    x2_shadow_enabled: bool = False
    mtab_enabled: bool = False
    pst_enabled: bool = False
    ssd_enabled: bool = False


class RuntimeConfig(BaseModel):
    """Runtime memory mode + per-mode feature override."""
    memory_mode: RuntimeMode = "moderate"
    moderate: ModeFeatures = Field(default_factory=lambda: ModeFeatures(
        ctrsp_lru_size=32,
        draft_model_path="",
        x2_shadow_enabled=False,
        mtab_enabled=False,
        pst_enabled=False,
        ssd_enabled=False,
    ))
    aggressive: ModeFeatures = Field(default_factory=lambda: ModeFeatures(
        ctrsp_lru_size=32,
        draft_model_path="",
        x2_shadow_enabled=True,
        mtab_enabled=True,
        pst_enabled=False,
        ssd_enabled=True,
    ))
    extreme: ModeFeatures = Field(default_factory=lambda: ModeFeatures(
        ctrsp_lru_size=16,
        draft_model_path="",
        x2_shadow_enabled=True,
        mtab_enabled=True,
        pst_enabled=True,
        ssd_enabled=True,
    ))

    def active_features(self) -> ModeFeatures:
        """有効モードに対応する ModeFeatures を返す."""
        return getattr(self, self.memory_mode)


# ---------------------------------------------------------------------------
# Top-level engine settings (composes all sections)
# ---------------------------------------------------------------------------

class ModelFamilyConfig(BaseModel):
    """v3 generalization: per-family behavior knobs.

    Default values keep Qwen3.5 behavior intact. Users plugging in other
    models (Gemma 4, Qwen 3.6, Llama 3 等) override `family` or leave
    "auto" for detection by `app.core.model_family.detect_family`.
    """
    family: str = "auto"           # "auto" | "qwen35" | "qwen36" | "qwen25" | "llama" | "mistral" | "gemma"
    main_path: str = ""            # Optional — if empty, use registry / runtime default
    draft_path: str = ""           # Optional override for RDMS draft
    # Per-family override (only used if family != auto)
    override_supports_thinking: bool | None = None
    override_is_hybrid: bool | None = None
    # Compatibility mode: if draft and main families differ, what to do
    draft_family_mismatch: Literal["warn", "refuse", "ignore"] = "warn"


class EngineSettings(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    speculative: SpeculativeConfig = Field(default_factory=SpeculativeConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    innovation: InnovationConfig = Field(default_factory=InnovationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    # v3 general: model family config (optional, default auto/qwen35)
    model: ModelFamilyConfig = Field(default_factory=ModelFamilyConfig)


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """Mirrors the ModelProfile fields from the SPEC."""

    name: str
    path: str
    modality: Literal["text", "image_text", "draft"]
    role: Literal["main", "sub_heavy", "sub_speed"]
    load_strategy: Literal["hot", "cold"] = "cold"
    supports_vision: bool = False
    estimated_model_ram_gb: float = 0.0
    max_context_tokens_hard: int = 32768
    max_output_tokens_hard: int = 4096


class SubModelConfig(BaseModel):
    mode: Literal["ephemeral", "persistent"] = "ephemeral"
    warm_ttl_sec: int = 180


# ---------------------------------------------------------------------------
# Singleton loaders
# ---------------------------------------------------------------------------

def _build_runtime_config(raw: dict) -> RuntimeConfig:
    """Parse [runtime] + [runtime.moderate/aggressive/extreme] sections."""
    runtime_raw = raw.get("runtime", {})
    # Top-level memory_mode
    rc = RuntimeConfig(memory_mode=runtime_raw.get("memory_mode", "moderate"))
    # Per-mode override (subsections)
    for mode in ("moderate", "aggressive", "extreme"):
        sub = runtime_raw.get(mode, {})
        if sub:
            current = getattr(rc, mode)
            updated = ModeFeatures(**{**current.model_dump(), **dict(sub)})
            setattr(rc, mode, updated)
    return rc


@functools.lru_cache(maxsize=1)
def get_settings() -> EngineSettings:
    """Load and return the singleton EngineSettings from engine.toml."""
    raw = _load_toml("engine.toml")
    return EngineSettings(
        server=ServerConfig(**raw.get("server", {})),
        engine=EngineConfig(**raw.get("engine", {})),
        memory=MemoryConfig(**raw.get("memory", {})),
        context=ContextConfig(**raw.get("context", {})),
        generation=GenerationConfig(**raw.get("generation", {})),
        cache=CacheConfig(**raw.get("cache", {})),
        speculative=SpeculativeConfig(**raw.get("speculative", {})),
        planner=PlannerConfig(**raw.get("planner", {})),
        innovation=InnovationConfig(**raw.get("innovation", {})),
        runtime=_build_runtime_config(raw),
        model=ModelFamilyConfig(**raw.get("model", {})),
    )


@functools.lru_cache(maxsize=1)
def get_model_configs() -> Dict[str, ModelConfig]:
    """Load and return a dict of model configs keyed by role name."""
    raw = _load_toml("models.toml")
    models_raw = raw.get("models", {})
    result: Dict[str, ModelConfig] = {}
    for key, values in models_raw.items():
        result[key] = ModelConfig(**values)
    return result


@functools.lru_cache(maxsize=1)
def get_sub_model_config() -> SubModelConfig:
    """Load and return the sub_model lifecycle config."""
    raw = _load_toml("models.toml")
    return SubModelConfig(**raw.get("sub_model", {}))
