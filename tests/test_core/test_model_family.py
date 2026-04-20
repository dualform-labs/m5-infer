"""Tests for model_family detection and profile registry."""
from __future__ import annotations
import pytest

from app.core.model_family import (
    ModelFamily, FAMILY_PROFILES, get_profile, detect_family, describe,
)


def test_family_enum_covers_known():
    assert ModelFamily.QWEN_3_5.value == "qwen35"
    assert ModelFamily.QWEN_3_6.value == "qwen36"
    assert ModelFamily.GEMMA.value == "gemma"
    assert ModelFamily.LLAMA.value == "llama"


def test_from_string_roundtrip():
    assert ModelFamily.from_string("qwen35") == ModelFamily.QWEN_3_5
    assert ModelFamily.from_string("QWEN35") == ModelFamily.QWEN_3_5
    assert ModelFamily.from_string("gemma") == ModelFamily.GEMMA
    assert ModelFamily.from_string("") == ModelFamily.UNKNOWN
    assert ModelFamily.from_string("nonsense") == ModelFamily.UNKNOWN


def test_profiles_complete():
    for fam in ModelFamily:
        prof = get_profile(fam)
        assert prof.family == fam


def test_qwen35_profile_is_hybrid_with_thinking():
    p = get_profile(ModelFamily.QWEN_3_5)
    assert p.is_hybrid is True
    assert p.has_gdn is True
    assert p.supports_thinking is True
    assert p.think_open_token == "<think>"
    assert p.think_close_token == "</think>"


def test_llama_profile_pure_transformer_no_thinking():
    p = get_profile(ModelFamily.LLAMA)
    assert p.is_hybrid is False
    assert p.has_gdn is False
    assert p.supports_thinking is False


def test_detect_path_qwen35():
    assert detect_family("mlx-community/Qwen3.5-9B-MLX-4bit") == ModelFamily.QWEN_3_5


def test_detect_path_qwen36():
    assert detect_family("mlx-community/Qwen3.6-7B-Instruct-4bit") == ModelFamily.QWEN_3_6


def test_detect_path_gemma():
    for p in ("google/gemma-4-9b-it", "mlx-community/gemma-3-4b", "gemma2-2b"):
        assert detect_family(p) == ModelFamily.GEMMA


def test_detect_path_llama():
    for p in ("meta-llama/Llama-3-8B", "Llama3-70B", "llama-2-7b"):
        assert detect_family(p) == ModelFamily.LLAMA


def test_detect_path_mistral():
    assert detect_family("mlx-community/Mistral-7B-v0.3-4bit") == ModelFamily.MISTRAL


def test_detect_empty_path_unknown():
    assert detect_family("") == ModelFamily.UNKNOWN


def test_detect_nonsense_unknown():
    assert detect_family("some-unknown-model") == ModelFamily.UNKNOWN


def test_describe_shape():
    d = describe(ModelFamily.QWEN_3_5)
    for key in ("family", "is_hybrid", "has_gdn", "supports_thinking",
                "think_open", "think_close", "mask_module", "notes"):
        assert key in d


def test_model_family_config_loadable():
    """EngineSettings can be built with or without [model] section."""
    from app.core.config import ModelFamilyConfig, EngineSettings
    # Default
    cfg = ModelFamilyConfig()
    assert cfg.family == "auto"
    assert cfg.draft_family_mismatch == "warn"
    # Explicit
    cfg2 = ModelFamilyConfig(family="gemma", main_path="google/gemma-4-9b")
    assert cfg2.family == "gemma"
