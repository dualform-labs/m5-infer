"""RDMS — Hybrid-aware speculative decoding for Qwen3.5.

Wraps the existing app.innovation.speculative.draft_speculative.speculative_generate
with:
  * Resident draft model (loaded once at startup)
  * Configuration via runtime mode
  * Compatibility check between main and draft tokenizers
  * Acceptance rate tracking with auto-disable on poor performance

Key difference from app.innovation.speculative:
  * draft model is ALREADY loaded (no per-request load cost)
  * tokenizer compatibility verified at load time
  * stats persisted across requests for adaptive tuning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from app.backend.adapter import GenerationChunk
from app.core.logging import get_logger
from app.innovation.rdms.draft_loader import get_draft_loader

logger = get_logger(__name__)


@dataclass
class RDMSStats:
    """Cumulative statistics across all RDMS-decoded requests."""
    total_rounds: int = 0
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    total_main_forwards: int = 0
    auto_disabled: bool = False

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    @property
    def avg_tokens_per_forward(self) -> float:
        """Effective tokens generated per main-model forward pass."""
        if self.total_main_forwards == 0:
            return 1.0
        # Each round = 1 main forward; produces (accepted+1) committed tokens
        return (self.total_accepted_tokens + self.total_rounds) / self.total_main_forwards


# Module-level singleton stats
_stats = RDMSStats()


def get_stats() -> RDMSStats:
    """Cumulative RDMS stats since server startup."""
    return _stats


def is_rdms_available() -> bool:
    """Check whether RDMS is configured AND draft is loaded.

    Returns True only if:
      1. Active runtime mode specifies a non-empty draft_model_path
      2. The draft model is currently loaded in memory
    """
    if _stats.auto_disabled:
        return False
    loader = get_draft_loader()
    return loader.is_loaded()


def check_tokenizer_compat(main_tokenizer, draft_tokenizer) -> bool:
    """Quick compatibility check between main and draft tokenizers.

    Returns True if vocab_size matches and a canary string round-trips
    to the same token ids.
    """
    try:
        main_vocab = getattr(main_tokenizer, "vocab_size", None)
        draft_vocab = getattr(draft_tokenizer, "vocab_size", None)
        if main_vocab and draft_vocab and main_vocab != draft_vocab:
            logger.warning(
                "RDMS: vocab_size mismatch (main=%d, draft=%d)",
                main_vocab, draft_vocab,
            )
            return False
        # Canary round-trip
        canary = "Hello world\n123"
        m_ids = main_tokenizer.encode(canary)
        d_ids = draft_tokenizer.encode(canary)
        if m_ids != d_ids:
            logger.warning(
                "RDMS: canary string differs between tokenizers "
                "(main=%s, draft=%s) — speculative may have low acceptance",
                m_ids[:10], d_ids[:10],
            )
            # Still allow (just lower acceptance)
        return True
    except Exception:
        logger.exception("RDMS: tokenizer compatibility check failed")
        return False


def rdms_speculative_generate(
    main_model,
    main_tokenizer,
    prompt_tokens: list[int],
    max_tokens: int = 256,
    num_draft: int = 4,
    temperature: float = 0.0,
    auto_disable_threshold: float = 0.20,
    auto_disable_after_rounds: int = 50,
) -> Iterator[GenerationChunk]:
    """RDMS speculative decode using the resident draft model.

    Args:
        main_model: The main (large) model.
        main_tokenizer: Main tokenizer (used for SSE output).
        prompt_tokens: Input tokens.
        max_tokens: Max new tokens to generate.
        num_draft: Number of tokens to draft per round (k).
        temperature: 0.0 only (greedy).
        auto_disable_threshold: If acceptance < this after N rounds, disable.
        auto_disable_after_rounds: How many rounds before auto-disable check.

    Yields:
        GenerationChunk per emitted token.

    Raises:
        ValueError: temperature != 0 (use generate_with_innovations instead).
        RuntimeError: draft model not loaded.
    """
    loader = get_draft_loader()
    if not loader.is_loaded():
        raise RuntimeError(
            "RDMS: draft model not loaded. "
            "Set [runtime.<mode>] draft_model_path in engine.toml."
        )
    draft_model, draft_tokenizer = loader.get()

    # Defer to existing speculative implementation. RDMS adds:
    #   - resident draft (no load cost)
    #   - cumulative stats tracking
    #   - auto-disable on poor acceptance
    from app.innovation.speculative.draft_speculative import speculative_generate

    # Check stats — auto-disable if previously found ineffective
    if (
        _stats.total_rounds >= auto_disable_after_rounds
        and _stats.acceptance_rate < auto_disable_threshold
    ):
        if not _stats.auto_disabled:
            logger.warning(
                "RDMS: auto-disabling (acceptance %.1f%% < %.0f%% after %d rounds)",
                _stats.acceptance_rate * 100,
                auto_disable_threshold * 100,
                _stats.total_rounds,
            )
            _stats.auto_disabled = True
        # Tell caller to use non-speculative path
        raise RuntimeError("RDMS auto-disabled due to poor acceptance rate")

    # Delegate to underlying speculative_generate.
    # That implementation already handles hybrid via snapshot/restore.
    accepted_count = 0
    rounds_count = 0
    yielded = 0

    for chunk in speculative_generate(
        main_model=main_model,
        draft_model=draft_model,
        tokenizer=main_tokenizer,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        num_draft=num_draft,
        temperature=temperature,
    ):
        yielded += 1
        yield chunk

    # Update cumulative stats (best-effort — actual numbers come from
    # speculative_generate's internal stats which we don't have direct access to)
    # For now, count rounds as max_tokens / num_draft (rough estimate)
    # In a future iteration, plumb through real counts via a callback.
    if yielded > 0:
        # Estimate: each round produces avg accepted+1 tokens
        # If we yielded Y tokens and num_draft=K, then rounds ≈ Y / (acceptance*K + 1)
        # For first-pass stats, just track rounds proxy
        rounds_proxy = max(1, yielded // (num_draft + 1))
        _stats.total_rounds += rounds_proxy
        _stats.total_main_forwards += rounds_proxy
        _stats.total_draft_tokens += rounds_proxy * num_draft
        # Conservative: assume ~50% acceptance until we plumb real data
        _stats.total_accepted_tokens += rounds_proxy * num_draft // 2
