"""Hardware profile detection for Apple Silicon Macs.

Detects chip family, variant, GPU/ANE cores, unified memory size,
and macOS version. Produces a structured `ChipProfile` used by
`app.core.auto_tune` to pick optimal engine settings at startup.

Design goals:
- Run cheaply at startup (all `sysctl` calls, single-digit ms)
- Fail-safe: never raise, fall back to a conservative default profile
- Publishable: no machine-specific identifiers (serial, MAC) are read
"""
from __future__ import annotations

import logging
import platform
import re
import subprocess
from dataclasses import dataclass, field
from typing import Literal, Optional

logger = logging.getLogger(__name__)


ChipFamily = Literal["M1", "M2", "M3", "M4", "M5", "Unknown"]
ChipVariant = Literal["Base", "Pro", "Max", "Ultra", "Unknown"]


@dataclass
class ChipProfile:
    """Snapshot of the current Mac's key capabilities."""

    family: ChipFamily = "Unknown"
    variant: ChipVariant = "Unknown"
    generation: int = 0              # 1..5 for M1..M5, 0 for unknown
    chip_name: str = ""              # e.g., "Apple M5 Pro"
    gpu_cores: int = 0               # detected or estimated
    ane_cores: int = 0               # estimated by family
    ane_tops: float = 0.0            # estimated TOPS
    memory_gb: float = 0.0           # unified memory
    memory_bandwidth_gbps: int = 0   # estimated
    performance_cores: int = 0
    efficiency_cores: int = 0
    macos_version: str = ""
    supports_metal_4: bool = False
    is_apple_silicon: bool = False
    warnings: list[str] = field(default_factory=list)

    def tier(self) -> Literal["entry", "standard", "high", "ultra"]:
        """Quick tier classification used by auto_tune."""
        if self.memory_bandwidth_gbps >= 800:
            return "ultra"
        if self.memory_bandwidth_gbps >= 250:
            return "high"
        if self.memory_bandwidth_gbps >= 100:
            return "standard"
        return "entry"

    def summary(self) -> str:
        return (
            f"{self.chip_name} | {self.memory_gb:.0f} GB unified | "
            f"GPU={self.gpu_cores} ANE={self.ane_cores} "
            f"({self.ane_tops:.0f} TOPS) | BW={self.memory_bandwidth_gbps} GB/s | "
            f"macOS {self.macos_version} | tier={self.tier()}"
        )


# ─────────────────────────────────────────────────────────
# Per-family defaults (bandwidth estimates per Apple specs)
# ─────────────────────────────────────────────────────────
# Memory bandwidth in GB/s. Sourced from Apple official Mac comparison
# and product specification pages (verified 2026-04). "Ultra" rows omitted
# where Apple never shipped that tier for a given generation.
_MEMORY_BANDWIDTH_GBPS = {
    # M1 (2020-2022): full lineup shipped
    ("M1", "Base"): 68, ("M1", "Pro"): 200, ("M1", "Max"): 400, ("M1", "Ultra"): 800,
    # M2 (2022-2023): full lineup shipped
    ("M2", "Base"): 100, ("M2", "Pro"): 200, ("M2", "Max"): 400, ("M2", "Ultra"): 800,
    # M3 (2023-2025): Ultra shipped March 2025 in Mac Studio
    ("M3", "Base"): 100, ("M3", "Pro"): 150, ("M3", "Max"): 400, ("M3", "Ultra"): 800,
    # M4 (2024-): NO Ultra variant — Apple did not ship M4 Ultra
    ("M4", "Base"): 120, ("M4", "Pro"): 273, ("M4", "Max"): 546,
    # M5 (2025-): NO Ultra variant — not released as of 2026-04
    ("M5", "Base"): 153, ("M5", "Pro"): 307, ("M5", "Max"): 614,
}

_ANE_CORES_BY_GEN = {1: 16, 2: 16, 3: 16, 4: 16, 5: 16}
# Apple claims: M5 ANE ≈ 256 TOPS (vs M4's ≈ 38 TOPS for LLM-optimized ops)
_ANE_TOPS_BY_GEN = {1: 11, 2: 15.8, 3: 18, 4: 38, 5: 256}

# Max GPU cores per (family, variant). Verified 2026-04 from Apple specs.
# Entries where Apple never shipped that variant are omitted.
_GPU_CORES_RANGE = {
    ("M1", "Base"): 8, ("M1", "Pro"): 16, ("M1", "Max"): 32, ("M1", "Ultra"): 64,
    ("M2", "Base"): 10, ("M2", "Pro"): 19, ("M2", "Max"): 38, ("M2", "Ultra"): 76,
    ("M3", "Base"): 10, ("M3", "Pro"): 18, ("M3", "Max"): 40, ("M3", "Ultra"): 80,
    # M4: no Ultra variant — Apple did not ship M4 Ultra.
    ("M4", "Base"): 10, ("M4", "Pro"): 20, ("M4", "Max"): 40,
    # M5: no Ultra variant as of 2026-04.
    ("M5", "Base"): 10, ("M5", "Pro"): 20, ("M5", "Max"): 40,
}


# ─────────────────────────────────────────────────────────
# sysctl helpers
# ─────────────────────────────────────────────────────────
def _sysctl(key: str, default: str = "") -> str:
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", key], stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip()
        return out or default
    except Exception:
        return default


def _sysctl_int(key: str, default: int = 0) -> int:
    s = _sysctl(key, "")
    if not s:
        return default
    try:
        return int(s)
    except ValueError:
        return default


# ─────────────────────────────────────────────────────────
# Parsing the chip brand
# ─────────────────────────────────────────────────────────
_BRAND_RE = re.compile(
    r"Apple\s+(M)(\d)\s*(Pro|Max|Ultra)?",
    re.IGNORECASE,
)


def _parse_brand(brand: str) -> tuple[ChipFamily, ChipVariant, int]:
    m = _BRAND_RE.search(brand or "")
    if not m:
        return "Unknown", "Unknown", 0
    gen = int(m.group(2))
    family_key = f"M{gen}"
    if family_key not in {"M1", "M2", "M3", "M4", "M5"}:
        family_key = "Unknown"
    variant_raw = (m.group(3) or "Base").title()
    if variant_raw not in {"Base", "Pro", "Max", "Ultra"}:
        variant_raw = "Base"
    return family_key, variant_raw, gen  # type: ignore


def _parse_macos_version() -> tuple[str, bool]:
    """Return (version_string, supports_metal_4).

    macOS 26.x is the 2026 release line; that is our minimum target.
    Metal 4 is formally advertised from macOS 26.0 onward.
    """
    ver = platform.mac_ver()[0] or ""
    supports_m4 = False
    try:
        major = int(ver.split(".")[0]) if ver else 0
        minor = int(ver.split(".")[1]) if ver and "." in ver else 0
        supports_m4 = (major >= 26)
        supports_26_4 = (major > 26) or (major == 26 and minor >= 4)
        if not supports_26_4 and major > 0:
            logger.warning(
                "macOS %s detected; recommended: 26.4 or newer for full feature set.",
                ver,
            )
    except Exception:
        pass
    return ver, supports_m4


# ─────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────
def detect_hardware() -> ChipProfile:
    """Detect the running Mac's ChipProfile. Never raises."""
    profile = ChipProfile()
    if platform.system() != "Darwin":
        profile.warnings.append(
            "Non-macOS system: m5-infer targets Apple Silicon and is not tested on other platforms."
        )
        return profile

    # Apple Silicon check: sysctl hw.optional.arm64 == 1
    is_arm = _sysctl_int("hw.optional.arm64", 0) == 1 or platform.machine() in ("arm64", "aarch64")
    profile.is_apple_silicon = is_arm
    if not is_arm:
        profile.warnings.append(
            "This Mac does not appear to be Apple Silicon. m5-infer requires M1 or newer."
        )
        return profile

    # Brand
    brand = _sysctl("machdep.cpu.brand_string", "")
    profile.chip_name = brand or "Apple Silicon (unknown)"
    fam, var, gen = _parse_brand(brand)
    profile.family = fam
    profile.variant = var
    profile.generation = gen

    # Cores
    profile.performance_cores = _sysctl_int("hw.perflevel0.physicalcpu", 0)
    profile.efficiency_cores = _sysctl_int("hw.perflevel1.physicalcpu", 0)

    # Memory
    mem_bytes = _sysctl_int("hw.memsize", 0)
    profile.memory_gb = mem_bytes / (1024 ** 3) if mem_bytes else 0.0

    # Derived: bandwidth, GPU/ANE estimates
    if profile.family != "Unknown" and profile.variant != "Unknown":
        bw = _MEMORY_BANDWIDTH_GBPS.get((profile.family, profile.variant))
        if bw is None:
            # Unknown combination (e.g., future M5/M4 Ultra). Use a conservative
            # per-variant fallback so the tier selection still works reasonably.
            fallback_by_variant = {"Base": 100, "Pro": 250, "Max": 500, "Ultra": 800}
            bw = fallback_by_variant.get(profile.variant, 100)
            profile.warnings.append(
                f"Unrecognized chip combination {profile.family} {profile.variant} — "
                f"using fallback bandwidth estimate {bw} GB/s. "
                "Please report this so the profile table can be updated."
            )
        profile.memory_bandwidth_gbps = bw
        profile.gpu_cores = _GPU_CORES_RANGE.get((profile.family, profile.variant), 0)
        profile.ane_cores = _ANE_CORES_BY_GEN.get(gen, 0)
        profile.ane_tops = _ANE_TOPS_BY_GEN.get(gen, 0.0)

    # macOS
    profile.macos_version, profile.supports_metal_4 = _parse_macos_version()

    # Minimum requirement checks
    if profile.generation and profile.generation < 1:
        profile.warnings.append("Chip generation could not be resolved.")
    if profile.memory_gb > 0 and profile.memory_gb < 16:
        profile.warnings.append(
            f"Unified memory {profile.memory_gb:.1f} GB is below the 16 GB minimum. "
            "9B-class 4-bit models will not load; consider smaller models (3B or less)."
        )
    elif 16 <= profile.memory_gb < 24:
        profile.warnings.append(
            f"Unified memory {profile.memory_gb:.0f} GB meets minimum. "
            "24 GB+ is recommended for 9B-class 4-bit models."
        )

    logger.info("hardware_profile: %s", profile.summary())
    for w in profile.warnings:
        logger.warning("hardware_profile: %s", w)
    return profile


__all__ = ["ChipProfile", "ChipFamily", "ChipVariant", "detect_hardware"]
