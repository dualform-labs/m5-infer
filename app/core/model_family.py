"""Model family detection and feature profile — v3 generalization layer.

v3 の innovation は元々 Qwen3.5 hybrid (24 GDN + 8 FA) を前提に書かれて
いますが、本モジュールで **model family を自動判別** し、異なるモデル
(Gemma 4, Qwen 3.6, Llama 3, Mistral 等) でも同じエンジンで扱えるよう
抽象化します。

設計原則:
1. **既存 Qwen3.5 挙動は変更しない** — detect が qwen35 なら従来と完全同一
2. **非対応 family は明示的に RuntimeError** — 黙って壊れない
3. **auto-detect** は HuggingFace config.json (architectures, hidden_size,
   num_hidden_layers) を読んで推定

追加手順 (開発者向け):
    1. `FAMILY_PROFILES` に新 family のエントリを追加
    2. mask adapter に該当 family の分岐を追加 (`app/backend/mask_adapter.py`)
    3. 必要なら think tokens / chat template / KV cache class を指定
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ModelFamily(str, Enum):
    """Known model families (more can be added without breaking existing logic)."""
    QWEN_3_5 = "qwen35"       # Hybrid (GDN + FA)、ネイティブ対応
    QWEN_3_6 = "qwen36"       # 想定 (実装時に調整)
    QWEN_2_5 = "qwen25"       # Pure transformer
    LLAMA = "llama"            # Pure transformer
    MISTRAL = "mistral"        # Pure transformer
    GEMMA = "gemma"            # Pure transformer (Gemma 2/3/4)
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, s: str) -> "ModelFamily":
        s = (s or "").lower().strip()
        for member in cls:
            if member.value == s:
                return member
        return cls.UNKNOWN


@dataclass
class FamilyProfile:
    """Per-family behavioral profile. `None` means auto-detect or not applicable."""
    family: ModelFamily
    # Architecture
    is_hybrid: bool = False              # 部分的に recurrent (GDN 等) を含む
    has_gdn: bool = False                # GatedDeltaNet 層を持つ (Qwen3.5)
    # Thinking mode
    supports_thinking: bool = False      # chat template で <think> を prefill するか
    think_open_token: Optional[str] = None
    think_close_token: Optional[str] = None
    # MLX mask module (dotted path)
    mask_module: Optional[str] = None    # e.g., "mlx_lm.models.qwen3_5"
    # Notes for humans
    notes: str = ""


FAMILY_PROFILES: dict[ModelFamily, FamilyProfile] = {
    ModelFamily.QWEN_3_5: FamilyProfile(
        family=ModelFamily.QWEN_3_5,
        is_hybrid=True,
        has_gdn=True,
        supports_thinking=True,
        think_open_token="<think>",
        think_close_token="</think>",
        mask_module="mlx_lm.models.qwen3_5",
        notes="v3 のネイティブ対応モデル。24 GDN + 8 FA の hybrid。",
    ),
    ModelFamily.QWEN_3_6: FamilyProfile(
        family=ModelFamily.QWEN_3_6,
        is_hybrid=True,     # 想定、mlx_lm の実装を確認のこと
        has_gdn=True,
        supports_thinking=True,
        think_open_token="<think>",
        think_close_token="</think>",
        mask_module="mlx_lm.models.qwen3_6",
        notes="Qwen 3.6。mlx_lm で対応完了後、module path を確認。",
    ),
    ModelFamily.QWEN_2_5: FamilyProfile(
        family=ModelFamily.QWEN_2_5,
        is_hybrid=False,
        has_gdn=False,
        supports_thinking=False,
        mask_module="mlx_lm.models.qwen2",
        notes="Pure transformer。speculative decoding が FAST PATH で効く。",
    ),
    ModelFamily.LLAMA: FamilyProfile(
        family=ModelFamily.LLAMA,
        is_hybrid=False,
        has_gdn=False,
        supports_thinking=False,
        mask_module="mlx_lm.models.llama",
        notes="Pure transformer。Llama 3/3.1/3.2/3.3 を想定。",
    ),
    ModelFamily.MISTRAL: FamilyProfile(
        family=ModelFamily.MISTRAL,
        is_hybrid=False,
        has_gdn=False,
        supports_thinking=False,
        mask_module="mlx_lm.models.mistral",
        notes="Pure transformer。SWA を持つものあり。",
    ),
    ModelFamily.GEMMA: FamilyProfile(
        family=ModelFamily.GEMMA,
        is_hybrid=False,
        has_gdn=False,
        supports_thinking=False,
        mask_module="mlx_lm.models.gemma3",  # gemma2/gemma3/gemma4 いずれか実装に合わせ
        notes="Gemma 2/3/4。Gemma 4 は mlx_lm が対応したら module 名を確認。",
    ),
    ModelFamily.UNKNOWN: FamilyProfile(
        family=ModelFamily.UNKNOWN,
        is_hybrid=False,
        has_gdn=False,
        supports_thinking=False,
        notes="Unknown 扱い。Innovation は最小機能のみ動作 (speculative off、think hooks off)。",
    ),
}


def get_profile(family: ModelFamily) -> FamilyProfile:
    return FAMILY_PROFILES[family]


# ─────────────────────────────────────────────────────────
# Auto-detection from HF config.json / model path
# ─────────────────────────────────────────────────────────
_PATH_HEURISTICS = [
    # (substring, family)
    ("qwen3.5", ModelFamily.QWEN_3_5),
    ("qwen3_5", ModelFamily.QWEN_3_5),
    ("qwen-3.5", ModelFamily.QWEN_3_5),
    ("qwen3.6", ModelFamily.QWEN_3_6),
    ("qwen3_6", ModelFamily.QWEN_3_6),
    ("qwen2.5", ModelFamily.QWEN_2_5),
    ("qwen2_5", ModelFamily.QWEN_2_5),
    ("qwen-2.5", ModelFamily.QWEN_2_5),
    ("llama-3", ModelFamily.LLAMA),
    ("llama3", ModelFamily.LLAMA),
    ("llama-2", ModelFamily.LLAMA),
    ("mistral", ModelFamily.MISTRAL),
    ("gemma-4", ModelFamily.GEMMA),
    ("gemma4", ModelFamily.GEMMA),
    ("gemma-3", ModelFamily.GEMMA),
    ("gemma3", ModelFamily.GEMMA),
    ("gemma-2", ModelFamily.GEMMA),
    ("gemma2", ModelFamily.GEMMA),
]


_ARCH_TO_FAMILY = {
    "qwen3_5forcausallm": ModelFamily.QWEN_3_5,
    "qwen3_6forcausallm": ModelFamily.QWEN_3_6,
    "qwen2forcausallm": ModelFamily.QWEN_2_5,
    "llamaforcausallm": ModelFamily.LLAMA,
    "mistralforcausallm": ModelFamily.MISTRAL,
    "gemma2forcausallm": ModelFamily.GEMMA,
    "gemma3forcausallm": ModelFamily.GEMMA,
    "gemma4forcausallm": ModelFamily.GEMMA,
}


def detect_family(
    model_path_or_config: str,
    *,
    prefer_hf_config: bool = True,
) -> ModelFamily:
    """Detect model family from HF path or local config.json.

    Priority:
      1. If `prefer_hf_config`, try reading `config.json`'s `architectures`
      2. Path-substring heuristic
      3. UNKNOWN

    Side effect: logger.info of detection result.
    """
    if not model_path_or_config:
        return ModelFamily.UNKNOWN

    # Try HF config.json
    if prefer_hf_config:
        try:
            config_path = _resolve_config_json(model_path_or_config)
            if config_path and config_path.exists():
                conf = json.loads(config_path.read_text())
                archs = conf.get("architectures", [])
                if archs:
                    arch_key = archs[0].lower()
                    if arch_key in _ARCH_TO_FAMILY:
                        fam = _ARCH_TO_FAMILY[arch_key]
                        logger.info(
                            "model_family: detected %s via config.json architectures=%s",
                            fam.value, archs,
                        )
                        return fam
        except Exception:
            pass

    # Path substring heuristic
    p = model_path_or_config.lower()
    for sub, fam in _PATH_HEURISTICS:
        if sub in p:
            logger.info("model_family: detected %s via path heuristic (%s)", fam.value, sub)
            return fam

    logger.warning(
        "model_family: could not detect family for %s (falling back to UNKNOWN)",
        model_path_or_config,
    )
    return ModelFamily.UNKNOWN


def _resolve_config_json(model_path_or_id: str) -> Optional[Path]:
    """Resolve a local `config.json` path from a path or HF id.

    For local paths, returns `<path>/config.json`. For HF ids, we don't
    download (keep the auto-detect lightweight) and return None; the caller
    falls back to heuristics.
    """
    p = Path(model_path_or_id)
    if p.is_dir():
        return p / "config.json"
    if p.exists() and p.suffix == ".json":
        return p
    return None


def describe(family: ModelFamily) -> dict:
    """Human-readable description for /health and logs."""
    prof = get_profile(family)
    return {
        "family": family.value,
        "is_hybrid": prof.is_hybrid,
        "has_gdn": prof.has_gdn,
        "supports_thinking": prof.supports_thinking,
        "think_open": prof.think_open_token,
        "think_close": prof.think_close_token,
        "mask_module": prof.mask_module,
        "notes": prof.notes,
    }


__all__ = [
    "ModelFamily",
    "FamilyProfile",
    "FAMILY_PROFILES",
    "get_profile",
    "detect_family",
    "describe",
]
