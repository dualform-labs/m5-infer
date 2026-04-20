"""Innovation feature flags.

Reads from engine.toml [innovation] section via the config system.
Provides quick boolean checks for each innovation technique.
"""

from __future__ import annotations
from app.core.config import get_settings


def get_innovation_config():
    """Get the innovation config section."""
    return get_settings().innovation


def is_n1_ctrsp_enabled() -> bool:
    """N1: Cross-Turn Recurrent State Persistence."""
    return get_innovation_config().n1_ctrsp_enabled


def is_x2_dpc_enabled() -> bool:
    """X2: Dynamic Precision Cascading."""
    return get_innovation_config().x2_dpc_enabled


def is_n3_ssee_enabled() -> bool:
    """N3: Self-Speculative Early Exit."""
    return get_innovation_config().n3_ssee_enabled


def is_n4_als_enabled() -> bool:
    """N4: Adaptive Layer Skipping."""
    return get_innovation_config().n4_als_enabled


def is_x4_context_fold_enabled() -> bool:
    """X4: Context Folding."""
    return get_innovation_config().x4_context_fold_enabled


def is_n5_erp_enabled() -> bool:
    """N5: Expert Route Prediction."""
    return get_innovation_config().n5_erp_enabled


def is_n6_pes_enabled() -> bool:
    """N6: Parallel Expert Streaming."""
    return get_innovation_config().n6_pes_enabled


def is_x5r_compiled_enabled() -> bool:
    """X5-R: Compiled Generation Pipeline."""
    return get_innovation_config().x5r_compiled_enabled


def get_dpc_confidence_threshold() -> float:
    return get_innovation_config().x2_dpc_confidence_threshold


def get_als_similarity_threshold() -> float:
    return get_innovation_config().n4_als_similarity_threshold


def get_ssee_min_acceptance_rate() -> float:
    return get_innovation_config().n3_ssee_min_acceptance_rate


def get_ssee_num_draft_tokens() -> int:
    return get_innovation_config().n3_ssee_num_draft_tokens


def get_pes_num_parallel() -> int:
    return get_innovation_config().n6_pes_num_parallel


def get_enabled_innovations() -> list[str]:
    """Return list of enabled innovation technique names."""
    cfg = get_innovation_config()
    enabled = []
    if cfg.n1_ctrsp_enabled: enabled.append("N1:CTRSP")
    if cfg.x2_dpc_enabled: enabled.append("X2:DPC")
    if cfg.n3_ssee_enabled: enabled.append("N3:SSEE")
    if cfg.n4_als_enabled: enabled.append("N4:ALS")
    if cfg.x4_context_fold_enabled: enabled.append("X4:ContextFold")
    if cfg.n5_erp_enabled: enabled.append("N5:ERP")
    if cfg.n6_pes_enabled: enabled.append("N6:PES")
    if cfg.x5r_compiled_enabled: enabled.append("X5R:Compiled")
    return enabled
