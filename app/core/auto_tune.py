"""Auto-tune engine settings based on detected hardware profile.

The same engine code runs from M1 base to M5 Ultra. This module picks
per-tier defaults so new users get near-optimal behavior without
touching `engine.toml`. Users can always override by setting
`[runtime] auto_tune = false` and choosing values manually.

Principles:
- **No quality changes.** Only tuning parameters that trade off speed vs.
  memory (kv_bits, chunk size, concurrency, wired_limit), never accuracy.
- **Conservative on smaller chips.** If memory bandwidth < 100 GB/s we
  cap concurrency and use smaller chunks to avoid swap pressure.
- **Aggressive on larger chips.** M4 Max / M5 Pro+ unlock parallelism,
  bigger prefill chunks, and (optionally) extra innovation layers.
- **Pure overrides.** Never raise — on any error, fall back to existing
  config values.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from app.core.hardware_profile import ChipProfile

logger = logging.getLogger(__name__)


@dataclass
class TuneOverrides:
    """Computed per-tier overrides. Fields are Optional so only set keys
    are applied to the live config; None = leave current value alone.
    """
    # generation
    kv_bits: Optional[int] = None
    progressive_chunk_size: Optional[int] = None
    default_max_output_tokens: Optional[int] = None
    max_thinking_tokens: Optional[int] = None
    # engine
    max_concurrent_requests: Optional[int] = None
    request_timeout_sec: Optional[int] = None
    # memory
    green_threshold_gb: Optional[float] = None
    yellow_threshold_gb: Optional[float] = None
    red_threshold_gb: Optional[float] = None
    total_budget_gb: Optional[float] = None
    # cache
    prefix_cache_max_entries: Optional[int] = None
    session_cache_max_entries: Optional[int] = None
    # innovation hints
    mtab_enabled: Optional[bool] = None
    pst_enabled: Optional[bool] = None
    ssd_enabled: Optional[bool] = None
    lookahead_k: Optional[int] = None
    # v1.0: bandwidth-adaptive speculative draft size
    rdms_num_draft: Optional[int] = None
    # v1.0: recommended wired/memory limit in MB (just a recommendation — see
    # server.py for actual application with safety checks)
    wired_limit_mb: Optional[int] = None
    # description string for logging
    rationale: str = ""

    def non_null_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()
                if v is not None and k != "rationale"}


# ─────────────────────────────────────────────────────────
# Per-tier policy
# ─────────────────────────────────────────────────────────
def _policy_entry(p: ChipProfile) -> TuneOverrides:
    """M1 base / 7-8 GPU cores / 68 GB/s bandwidth. Very conservative."""
    mem = p.memory_gb or 16.0
    return TuneOverrides(
        kv_bits=4,
        progressive_chunk_size=512,
        default_max_output_tokens=4096,
        max_thinking_tokens=1024,
        max_concurrent_requests=1,
        request_timeout_sec=600,
        total_budget_gb=mem,
        green_threshold_gb=min(mem * 0.6, 10.0),
        yellow_threshold_gb=min(mem * 0.75, 12.0),
        red_threshold_gb=min(mem * 0.9, 14.0),
        prefix_cache_max_entries=16,
        session_cache_max_entries=4,
        mtab_enabled=False,
        pst_enabled=False,
        ssd_enabled=False,
        lookahead_k=2,
        rationale=(
            "entry tier: M1 base / low-bandwidth hardware. "
            "Conservative chunk + small caches to avoid memory pressure."
        ),
    )


def _policy_standard(p: ChipProfile) -> TuneOverrides:
    """M1 Pro / M2 base / M3-M5 base. ~100-200 GB/s."""
    mem = p.memory_gb or 16.0
    return TuneOverrides(
        kv_bits=4,
        progressive_chunk_size=1024,
        default_max_output_tokens=8192,
        max_thinking_tokens=1536,
        max_concurrent_requests=1,
        request_timeout_sec=600,
        total_budget_gb=mem,
        green_threshold_gb=mem * 0.7,
        yellow_threshold_gb=mem * 0.82,
        red_threshold_gb=mem * 0.92,
        prefix_cache_max_entries=32,
        session_cache_max_entries=8,
        mtab_enabled=False,
        pst_enabled=False,
        ssd_enabled=False,
        lookahead_k=4,
        rationale=(
            "standard tier: base/Pro Apple Silicon. "
            "Default innovations on, 4-bit KV for long context."
        ),
    )


def _policy_high(p: ChipProfile) -> TuneOverrides:
    """M-Pro / Max class. 250-550 GB/s. Unlock parallelism."""
    mem = p.memory_gb or 32.0
    return TuneOverrides(
        kv_bits=4,
        progressive_chunk_size=2048,
        default_max_output_tokens=16384,
        max_thinking_tokens=4096,
        max_concurrent_requests=2,
        request_timeout_sec=900,
        total_budget_gb=mem,
        green_threshold_gb=mem * 0.7,
        yellow_threshold_gb=mem * 0.82,
        red_threshold_gb=mem * 0.92,
        prefix_cache_max_entries=64,
        session_cache_max_entries=16,
        mtab_enabled=True,
        pst_enabled=True,
        ssd_enabled=True,
        lookahead_k=4,
        rationale=(
            "high tier: Pro/Max class. "
            "MTAB+SSD activated; 2-way concurrency; 2K chunks."
        ),
    )


def _policy_ultra(p: ChipProfile) -> TuneOverrides:
    """Ultra-class SoC (M1/M2/M3/M4 Ultra). 800+ GB/s, enterprise tier.
    M5 Ultra is not yet released; when Apple ships it the same policy
    applies automatically via the bandwidth-based tier classification.
    """
    mem = p.memory_gb or 128.0
    return TuneOverrides(
        kv_bits=4,
        progressive_chunk_size=4096,
        default_max_output_tokens=32768,
        max_thinking_tokens=8192,
        max_concurrent_requests=4,
        request_timeout_sec=1800,
        total_budget_gb=mem,
        green_threshold_gb=mem * 0.65,
        yellow_threshold_gb=mem * 0.8,
        red_threshold_gb=mem * 0.9,
        prefix_cache_max_entries=128,
        session_cache_max_entries=32,
        mtab_enabled=True,
        pst_enabled=True,
        ssd_enabled=True,
        lookahead_k=4,
        rationale=(
            "ultra tier: Ultra-class SoC. "
            "All innovations, 4-way concurrency, large buffers."
        ),
    )


_TIER_POLICY = {
    "entry": _policy_entry,
    "standard": _policy_standard,
    "high": _policy_high,
    "ultra": _policy_ultra,
}


def _concurrent_from_gpu_cores(gpu_cores: int, tier: str) -> int:
    """Refine max_concurrent_requests based on GPU core count.

    Bandwidth alone is coarse. A 20-core GPU can serve ~2 concurrent
    generations, a 40-core GPU ~4. Keep tier policy as a floor so we
    never exceed memory pressure on low-memory machines.
    """
    if gpu_cores >= 60:
        return 6
    if gpu_cores >= 30:
        return 4
    if gpu_cores >= 16:
        return 2
    return 1


def _rdms_k_from_bandwidth(bw_gbps: int) -> int:
    """Choose RDMS draft size from memory bandwidth.

    Memory-bandwidth-bound decode benefits from more draft tokens per
    main forward pass as BW rises. Balanced against draft computation
    cost (small-model overhead) — empirically:
      < 100 GB/s  → 3 (small BW, cheap draft)
      100-250     → 4 (typical base/pro)
      250-500     → 6 (pro/max)
      500+        → 8 (max/ultra)
    """
    if bw_gbps >= 500:
        return 8
    if bw_gbps >= 250:
        return 6
    if bw_gbps >= 100:
        return 4
    return 3


def _default_headroom_gb(memory_gb: float) -> float:
    """Per-tier Metal headroom (for OS + other processes).

    v1.1.2 tightened these after 24 GB users hit "Insufficient Memory" on
    18.6 GB models that physically would fit on the 24 GB Mac: the previous
    default left 6 GB headroom, too conservative. New scale assumes the
    machine is primarily a dev box where the user can close other apps
    before running large models.
    """
    if memory_gb <= 16:   return 4.0   # was 6 → 10 GB limit on 16 GB
    if memory_gb <= 24:   return 4.0   # was 6 → 20 GB limit on 24 GB (fits 35B A3B)
    if memory_gb <= 32:   return 5.0   # was 6 → 27 GB on 32 GB
    if memory_gb <= 64:   return 6.0   # was 8 → 42 GB on 48, 58 GB on 64
    return 7.0                          # was 8 → 121 GB on 128, 185 GB on 192


def _wired_limit_from_memory(
    memory_gb: float,
    headroom_override_gb: Optional[float] = None,
    absolute_limit_gb: Optional[float] = None,
) -> Optional[int]:
    """Recommend Metal wired-memory limit in MB.

    Priority (highest first):
      1. ``absolute_limit_gb`` — engine.toml ``[memory] metal_limit_gb`` or
         ``$M5_INFER_METAL_LIMIT_GB``. Trusted verbatim (clamped to RAM).
      2. ``headroom_override_gb`` — engine.toml ``[memory] headroom_gb`` or
         ``$M5_INFER_METAL_HEADROOM_GB``.
      3. Per-tier default (:func:`_default_headroom_gb`).

    Returns None when memory is below 16 GB (inference of interesting
    models at 4-bit already doesn't fit comfortably).
    """
    if absolute_limit_gb is not None:
        limit = max(4.0, min(float(absolute_limit_gb), memory_gb - 1.0))
        return int(limit * 1024)
    if memory_gb < 16:
        return None
    headroom_gb = (
        float(headroom_override_gb)
        if headroom_override_gb is not None
        else _default_headroom_gb(memory_gb)
    )
    limit_gb = max(8.0, memory_gb - headroom_gb)
    return int(limit_gb * 1024)


def compute_overrides(profile: ChipProfile) -> TuneOverrides:
    """Pick the tier policy based on the detected profile, then refine
    concurrency and speculative draft size from GPU cores / bandwidth.
    """
    tier = profile.tier()
    policy = _TIER_POLICY.get(tier, _policy_standard)
    ov = policy(profile)

    # Refine based on per-chip specifics
    if profile.gpu_cores > 0:
        refined = _concurrent_from_gpu_cores(profile.gpu_cores, tier)
        # Floor at tier policy to keep small-memory machines safe
        if ov.max_concurrent_requests is None or refined > ov.max_concurrent_requests:
            ov.max_concurrent_requests = refined

    if profile.memory_bandwidth_gbps > 0:
        ov.rdms_num_draft = _rdms_k_from_bandwidth(profile.memory_bandwidth_gbps)

    wired_mb = _wired_limit_from_memory(profile.memory_gb)
    if wired_mb:
        ov.wired_limit_mb = wired_mb

    logger.info(
        "auto_tune: tier=%s chip=%s gpu=%d bw=%d — %s",
        tier, profile.chip_name or "unknown",
        profile.gpu_cores, profile.memory_bandwidth_gbps, ov.rationale,
    )
    return ov


# ─────────────────────────────────────────────────────────
# Application to live EngineSettings (non-destructive)
# ─────────────────────────────────────────────────────────
def apply_overrides(settings, overrides: TuneOverrides) -> list[str]:
    """Apply overrides to a mutable EngineSettings instance.

    Returns a list of change descriptions for logging. The function
    only sets attributes that already exist on the target object.
    """
    changed: list[str] = []
    mapping = {
        ("generation", "kv_bits"): overrides.kv_bits,
        ("generation", "default_max_output_tokens"): overrides.default_max_output_tokens,
        ("generation", "max_thinking_tokens"): overrides.max_thinking_tokens,
        ("generation", "lookahead_k"): overrides.lookahead_k,
        ("engine", "max_concurrent_requests"): overrides.max_concurrent_requests,
        ("engine", "request_timeout_sec"): overrides.request_timeout_sec,
        ("memory", "total_budget_gb"): overrides.total_budget_gb,
        ("memory", "green_threshold_gb"): overrides.green_threshold_gb,
        ("memory", "yellow_threshold_gb"): overrides.yellow_threshold_gb,
        ("memory", "red_threshold_gb"): overrides.red_threshold_gb,
        ("cache", "prefix_cache_max_entries"): overrides.prefix_cache_max_entries,
        ("cache", "session_cache_max_entries"): overrides.session_cache_max_entries,
        # v1.0: innovation overrides
        ("innovation", "rdms_num_draft"): overrides.rdms_num_draft,
    }

    for (section_name, attr), value in mapping.items():
        if value is None:
            continue
        section = getattr(settings, section_name, None)
        if section is None or not hasattr(section, attr):
            continue
        current = getattr(section, attr)
        if current != value:
            try:
                setattr(section, attr, value)
                changed.append(f"{section_name}.{attr}: {current} → {value}")
            except Exception:
                pass
    if changed:
        logger.info("auto_tune: applied %d override(s): %s", len(changed), "; ".join(changed))
    else:
        logger.info("auto_tune: no overrides needed (config already matches tier)")
    return changed


__all__ = ["TuneOverrides", "compute_overrides", "apply_overrides"]
