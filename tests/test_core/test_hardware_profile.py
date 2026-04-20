"""Tests for hardware_profile + auto_tune."""
from __future__ import annotations
import platform

from app.core.hardware_profile import (
    ChipProfile, detect_hardware, _parse_brand,
)
from app.core.auto_tune import (
    TuneOverrides, compute_overrides, apply_overrides,
)


def test_chip_profile_default_ok():
    p = ChipProfile()
    assert p.family == "Unknown"
    assert p.tier() == "entry"


def test_parse_brand_m5():
    fam, var, gen = _parse_brand("Apple M5")
    assert fam == "M5" and var == "Base" and gen == 5


def test_parse_brand_m4_pro():
    fam, var, gen = _parse_brand("Apple M4 Pro")
    assert fam == "M4" and var == "Pro" and gen == 4


def test_parse_brand_m1_ultra():
    fam, var, gen = _parse_brand("Apple M1 Ultra")
    assert fam == "M1" and var == "Ultra" and gen == 1


def test_parse_brand_intel_unknown():
    fam, var, gen = _parse_brand("Intel Core i7")
    assert fam == "Unknown"


def test_detect_hardware_on_this_machine():
    """Smoke test: detection must not raise on any platform."""
    p = detect_hardware()
    assert isinstance(p, ChipProfile)
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        assert p.is_apple_silicon is True
        assert p.macos_version != ""


def test_tier_boundaries():
    p = ChipProfile(memory_bandwidth_gbps=50); assert p.tier() == "entry"
    p = ChipProfile(memory_bandwidth_gbps=150); assert p.tier() == "standard"
    p = ChipProfile(memory_bandwidth_gbps=300); assert p.tier() == "high"
    p = ChipProfile(memory_bandwidth_gbps=900); assert p.tier() == "ultra"


def test_compute_overrides_entry():
    p = ChipProfile(family="M1", variant="Base", memory_gb=16,
                     memory_bandwidth_gbps=68)
    ov = compute_overrides(p)
    assert ov.max_concurrent_requests == 1
    assert ov.progressive_chunk_size == 512
    assert ov.kv_bits == 4


def test_compute_overrides_standard():
    p = ChipProfile(family="M5", variant="Base", memory_gb=24,
                     memory_bandwidth_gbps=120)
    ov = compute_overrides(p)
    assert ov.max_concurrent_requests == 1
    assert ov.progressive_chunk_size == 1024
    assert ov.lookahead_k == 4


def test_compute_overrides_high():
    p = ChipProfile(family="M5", variant="Pro", memory_gb=48,
                     memory_bandwidth_gbps=273)
    ov = compute_overrides(p)
    assert ov.max_concurrent_requests == 2
    assert ov.mtab_enabled is True
    assert ov.progressive_chunk_size == 2048


def test_compute_overrides_ultra():
    p = ChipProfile(family="M5", variant="Ultra", memory_gb=192,
                     memory_bandwidth_gbps=1092)
    ov = compute_overrides(p)
    assert ov.max_concurrent_requests == 4
    assert ov.progressive_chunk_size == 4096


def test_apply_overrides_noop_on_missing_attrs():
    """Object with no known sections: apply_overrides must not raise."""
    class Empty:
        pass
    changed = apply_overrides(Empty(), TuneOverrides(kv_bits=8))
    assert changed == []


def test_apply_overrides_changes_existing_attrs():
    from app.core.config import EngineSettings
    settings = EngineSettings()
    ov = TuneOverrides(
        progressive_chunk_size=2048,  # ignored (no such attr on engine)
        max_concurrent_requests=4,
        kv_bits=8,
    )
    changed = apply_overrides(settings, ov)
    assert settings.engine.max_concurrent_requests == 4
    assert settings.generation.kv_bits == 8
    assert len(changed) >= 2


def test_engine_auto_tune_default_true():
    from app.core.config import EngineConfig
    assert EngineConfig().auto_tune is True


# ─────────────────────────────────────────────────────────
# v1.0: GPU-cores aware concurrency + bandwidth-adaptive num_draft
# ─────────────────────────────────────────────────────────
def test_concurrency_scales_with_gpu_cores():
    """Higher GPU core count → higher concurrency cap."""
    from app.core.auto_tune import _concurrent_from_gpu_cores
    assert _concurrent_from_gpu_cores(8, "entry") == 1
    assert _concurrent_from_gpu_cores(10, "standard") == 1
    assert _concurrent_from_gpu_cores(16, "high") == 2
    assert _concurrent_from_gpu_cores(20, "high") == 2
    assert _concurrent_from_gpu_cores(32, "ultra") == 4
    assert _concurrent_from_gpu_cores(40, "ultra") == 4
    assert _concurrent_from_gpu_cores(64, "ultra") == 6
    assert _concurrent_from_gpu_cores(80, "ultra") == 6


def test_rdms_k_scales_with_bandwidth():
    from app.core.auto_tune import _rdms_k_from_bandwidth
    assert _rdms_k_from_bandwidth(68) == 3   # M1 base
    assert _rdms_k_from_bandwidth(153) == 4  # M5 base
    assert _rdms_k_from_bandwidth(307) == 6  # M5 Pro
    assert _rdms_k_from_bandwidth(614) == 8  # M5 Max
    assert _rdms_k_from_bandwidth(800) == 8  # M3 Ultra


def test_wired_limit_respects_headroom():
    from app.core.auto_tune import _wired_limit_from_memory
    # Under 16 GB → no recommendation
    assert _wired_limit_from_memory(8) is None
    # 24 GB → 24 - 6 = 18 GB = 18432 MB
    assert _wired_limit_from_memory(24) == 18 * 1024
    # 64 GB (>32) → 64 - 8 = 56 GB
    assert _wired_limit_from_memory(64) == 56 * 1024


def test_compute_overrides_includes_rdms_k():
    p = ChipProfile(family="M5", variant="Max", memory_gb=48,
                     memory_bandwidth_gbps=614, gpu_cores=40)
    ov = compute_overrides(p)
    assert ov.rdms_num_draft == 8
    assert ov.max_concurrent_requests == 4
    assert ov.wired_limit_mb is not None


def test_compute_overrides_m1_base_conservative():
    p = ChipProfile(family="M1", variant="Base", memory_gb=16,
                     memory_bandwidth_gbps=68, gpu_cores=8)
    ov = compute_overrides(p)
    assert ov.rdms_num_draft == 3
    assert ov.max_concurrent_requests == 1


def test_compute_overrides_m5_base_standard():
    p = ChipProfile(family="M5", variant="Base", memory_gb=24,
                     memory_bandwidth_gbps=153, gpu_cores=10)
    ov = compute_overrides(p)
    assert ov.rdms_num_draft == 4
    assert ov.max_concurrent_requests == 1


def test_compute_overrides_m3_ultra_enterprise():
    p = ChipProfile(family="M3", variant="Ultra", memory_gb=192,
                     memory_bandwidth_gbps=800, gpu_cores=80)
    ov = compute_overrides(p)
    assert ov.rdms_num_draft == 8
    assert ov.max_concurrent_requests == 6   # 80 cores → 6 concurrent
    assert ov.wired_limit_mb == (192 - 8) * 1024


def test_rdms_num_draft_flows_through_innovation_config():
    from app.core.config import InnovationConfig
    cfg = InnovationConfig()
    assert hasattr(cfg, "rdms_num_draft")
    assert cfg.rdms_num_draft == 4

    from app.core.config import EngineSettings
    from app.core.auto_tune import TuneOverrides, apply_overrides
    settings = EngineSettings()
    changed = apply_overrides(settings, TuneOverrides(rdms_num_draft=6))
    assert settings.innovation.rdms_num_draft == 6
    assert any("rdms_num_draft" in c for c in changed)
