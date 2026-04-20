"""X4: Context Folding — Shared KV with position-aware delta computation.

For Qwen3.5, only the 8 Full Attention layers use KV cache.
The 24 GatedDeltaNet layers use recurrent state (no KV to fold).
"""

from __future__ import annotations
from dataclasses import dataclass
import mlx.core as mx
from app.innovation.x4_context_fold.schema_analyzer import FoldAnalysis, SchemaTemplate
from app.core.logging import get_logger

logger = get_logger(__name__)

@dataclass
class FoldedPrompt:
    """A prompt with folded tool schemas."""
    # Full token sequence (with schemas expanded)
    full_tokens: list[int]
    # Mapping: (start_pos, end_pos) for each schema in the full sequence
    schema_positions: list[tuple[int, int]]
    # Template token range that can be shared
    shared_prefix_range: tuple[int, int] | None  # (start, end) in full_tokens
    shared_suffix_range: tuple[int, int] | None
    # Whether folding is active
    is_folded: bool
    # Analysis results
    analysis: FoldAnalysis


def prepare_folded_prompt(
    prompt_tokens: list[int],
    tools: list[dict] | None,
    tokenizer,
    analysis: FoldAnalysis | None = None,
) -> FoldedPrompt:
    """Prepare a prompt with context folding metadata.

    This doesn't modify the token sequence itself — instead it provides
    metadata that the cache manager can use to share KV entries
    for the common template portions of tool schemas.

    In Phase 3, this is metadata-only. Actual KV sharing will be
    integrated with the cache manager when the model's prefill loop
    is customized.
    """
    if analysis is None or not analysis.worth_folding:
        return FoldedPrompt(
            full_tokens=prompt_tokens,
            schema_positions=[],
            shared_prefix_range=None,
            shared_suffix_range=None,
            is_folded=False,
            analysis=analysis or FoldAnalysis(
                templates=[], total_tokens=len(prompt_tokens),
                shared_tokens=0, unique_tokens=len(prompt_tokens),
                fold_ratio=0.0, worth_folding=False,
            ),
        )

    # For now, provide the folding metadata without modifying inference
    # The actual KV sharing will be done in a later integration step
    # when we hook into the model's forward pass

    logger.info(
        "Context Folding prepared: %d schemas, fold_ratio=%.2f, "
        "saving ~%d shared tokens",
        analysis.templates[0].schema_count if analysis.templates else 0,
        analysis.fold_ratio,
        analysis.shared_tokens,
    )

    return FoldedPrompt(
        full_tokens=prompt_tokens,
        schema_positions=[],  # Will be populated in integration step
        shared_prefix_range=None,
        shared_suffix_range=None,
        is_folded=True,
        analysis=analysis,
    )
