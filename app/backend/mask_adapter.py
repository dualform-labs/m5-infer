"""Attention mask adapter — family-aware wrapper for custom_forward.

v3 の `custom_forward()` は元々 `mlx_lm.models.qwen3_5` から
`create_attention_mask` / `create_ssm_mask` を直接 import していました。
他モデルへの汎用化のため、本 adapter で family に応じて正しい module を
動的 import します。

Qwen3.5 の挙動は完全に同一 — `get_masks(family=QWEN_3_5)` で元の実装と
1:1 対応します。
"""
from __future__ import annotations
import importlib
import logging
from dataclasses import dataclass
from typing import Any, Optional

from app.core.model_family import ModelFamily, FamilyProfile, get_profile

logger = logging.getLogger(__name__)


@dataclass
class MaskSet:
    """A set of masks produced for a forward pass.

    Pure transformer: only `fa_mask` is set. `ssm_mask` is None.
    Hybrid (GDN + FA): both are set.
    """
    fa_mask: Any = None
    ssm_mask: Any = None


_module_cache: dict[str, Any] = {}


def _import_mask_module(module_path: str):
    if module_path in _module_cache:
        return _module_cache[module_path]
    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        logger.warning("mask_adapter: module %s not importable (%s)", module_path, exc)
        return None
    _module_cache[module_path] = mod
    return mod


def get_masks(
    family: ModelFamily,
    inner_model,
    hidden_states,
    cache: list,
) -> MaskSet:
    """Build attention / SSM masks for the family.

    For Qwen3.5 this produces exactly the same output as the original
    direct-import code. For pure transformer families, only fa_mask is
    populated. For unsupported families, raises RuntimeError with a
    helpful message.
    """
    profile = get_profile(family)

    if family == ModelFamily.UNKNOWN:
        # Best effort: try Qwen3.5 module (backward-compat path for legacy configs)
        try:
            return _qwen35_masks(inner_model, hidden_states, cache)
        except Exception:
            raise RuntimeError(
                "mask_adapter: unknown model family, no mask module available. "
                "Set [model] family in engine.toml explicitly."
            )

    if profile.mask_module is None:
        raise RuntimeError(f"mask_adapter: family {family.value} has no mask_module configured")

    mod = _import_mask_module(profile.mask_module)
    if mod is None:
        raise RuntimeError(
            f"mask_adapter: module {profile.mask_module} not importable; "
            f"install matching mlx_lm version or add family profile"
        )

    # Family-specific mask construction
    if family in (ModelFamily.QWEN_3_5, ModelFamily.QWEN_3_6):
        return _qwen_hybrid_masks(mod, inner_model, hidden_states, cache)
    if family in (ModelFamily.QWEN_2_5, ModelFamily.LLAMA, ModelFamily.MISTRAL, ModelFamily.GEMMA):
        return _pure_transformer_masks(mod, inner_model, hidden_states, cache)

    raise RuntimeError(f"mask_adapter: unhandled family {family}")


def _qwen_hybrid_masks(mod, inner_model, hidden_states, cache: list) -> MaskSet:
    """Qwen 3.5/3.6 hybrid: FA + SSM masks."""
    create_attention_mask = getattr(mod, "create_attention_mask", None)
    create_ssm_mask = getattr(mod, "create_ssm_mask", None)
    if create_attention_mask is None or create_ssm_mask is None:
        raise RuntimeError(
            f"mask_adapter: {mod.__name__} missing create_attention_mask or create_ssm_mask"
        )
    fa_mask = create_attention_mask(hidden_states, cache[inner_model.fa_idx])
    ssm_mask = create_ssm_mask(hidden_states, cache[inner_model.ssm_idx])
    return MaskSet(fa_mask=fa_mask, ssm_mask=ssm_mask)


def _qwen35_masks(inner_model, hidden_states, cache: list) -> MaskSet:
    """Direct Qwen3.5 path (legacy fallback)."""
    from mlx_lm.models.qwen3_5 import create_attention_mask, create_ssm_mask
    fa_mask = create_attention_mask(hidden_states, cache[inner_model.fa_idx])
    ssm_mask = create_ssm_mask(hidden_states, cache[inner_model.ssm_idx])
    return MaskSet(fa_mask=fa_mask, ssm_mask=ssm_mask)


def _pure_transformer_masks(mod, inner_model, hidden_states, cache: list) -> MaskSet:
    """Pure transformer families: only a causal mask is required.

    mlx_lm の pure transformer 実装は `create_attention_mask` を備える
    (または `mlx_lm.models.base`)。最初の FA layer の cache を使う。
    """
    # Try module-level function first
    create_attention_mask = getattr(mod, "create_attention_mask", None)
    if create_attention_mask is None:
        # Fallback to mlx_lm.models.base (newer mlx_lm versions)
        try:
            from mlx_lm.models.base import create_attention_mask as _cam
            create_attention_mask = _cam
        except Exception:
            create_attention_mask = None
    if create_attention_mask is None:
        raise RuntimeError(
            f"mask_adapter: {mod.__name__} has no create_attention_mask; "
            "check mlx_lm version"
        )
    # Pure transformer: use the first (and only) cache entry
    fa_cache = cache[0] if cache else None
    fa_mask = create_attention_mask(hidden_states, fa_cache) if fa_cache is not None else None
    return MaskSet(fa_mask=fa_mask, ssm_mask=None)


__all__ = ["MaskSet", "get_masks"]
