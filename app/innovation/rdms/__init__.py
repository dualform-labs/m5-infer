"""RDMS — Resident Draft Model Speculative.

Hybrid-architecture aware speculative decoding that keeps a small draft model
resident in memory at startup. The draft predicts k tokens; the main model
verifies all k in a single batched forward pass; longest matching prefix is
accepted (lossless via greedy argmax matching).

Designed for Qwen3.5 hybrid (24 GDN + 8 FA) with snapshot-based rollback.

Activation: set `[runtime.<mode>] draft_model_path = "..."` in engine.toml.
"""

from app.innovation.rdms.draft_loader import DraftModelLoader
from app.innovation.rdms.hybrid_speculative import (
    rdms_speculative_generate,
    is_rdms_available,
)

__all__ = [
    "DraftModelLoader",
    "rdms_speculative_generate",
    "is_rdms_available",
]
