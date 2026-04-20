"""Context redundancy detector for long system prompts.

Scans a system prompt for:
  - Repeated substrings (exact duplicates)
  - Highly similar blocks (tool schemas with same structure)
  - Redundant rules / instructions

Reports redundancy ratio and identifies candidate blocks that could be
deduplicated by the caller for faster TTFT on first turn.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RedundancyReport:
    total_tokens: int
    unique_tokens: int
    redundancy_ratio: float  # 0.0 = no redundancy, 1.0 = all duplicated
    duplicate_ngrams: list[tuple[tuple[int, ...], int]]  # (ngram, count)
    recommendations: list[str]


def analyze_prompt_redundancy(
    prompt_tokens: list[int],
    ngram_size: int = 20,
    min_duplicate_count: int = 3,
) -> RedundancyReport:
    """Analyze a token sequence for redundant n-grams.

    Looks for token-level n-grams that repeat in the prompt.
    Typical offenders in agent system prompts:
      - Tool schemas with shared boilerplate
      - Repeated rule statements
      - Example patterns used multiple times

    Args:
        prompt_tokens: Tokenized system prompt.
        ngram_size: Length of n-grams to check (in tokens).
        min_duplicate_count: Report n-grams that repeat at least this many times.
    """
    if len(prompt_tokens) < ngram_size:
        return RedundancyReport(
            total_tokens=len(prompt_tokens),
            unique_tokens=len(prompt_tokens),
            redundancy_ratio=0.0,
            duplicate_ngrams=[],
            recommendations=[],
        )

    # Count n-gram occurrences
    ngram_counts: dict[tuple[int, ...], int] = defaultdict(int)
    for i in range(len(prompt_tokens) - ngram_size + 1):
        ngram = tuple(prompt_tokens[i : i + ngram_size])
        ngram_counts[ngram] += 1

    # Find duplicates
    duplicates = [
        (ng, count) for ng, count in ngram_counts.items() if count >= min_duplicate_count
    ]
    duplicates.sort(key=lambda x: -x[1])  # Most frequent first

    # Estimate redundant tokens (conservative: don't double-count overlapping ngrams)
    # Rough estimate: each extra occurrence after the first is redundant.
    redundant_token_count = sum(
        (count - 1) * ngram_size for _, count in duplicates
    )
    redundant_token_count = min(redundant_token_count, len(prompt_tokens) // 2)  # cap

    unique_tokens = len(prompt_tokens) - redundant_token_count
    redundancy_ratio = redundant_token_count / len(prompt_tokens)

    # Recommendations
    recommendations = []
    if redundancy_ratio > 0.3:
        recommendations.append(
            f"High redundancy detected ({redundancy_ratio:.0%}). Consider consolidating "
            f"repeated rules or tool schemas."
        )
    if len(duplicates) > 10:
        recommendations.append(
            f"{len(duplicates)} repeating {ngram_size}-token patterns found. "
            "Tool schemas may share boilerplate."
        )
    if len(prompt_tokens) > 4096 and redundancy_ratio > 0.15:
        recommendations.append(
            f"System prompt is {len(prompt_tokens)} tokens with {redundancy_ratio:.0%} redundancy. "
            "Deduplication could save ~{:.0f} tokens per turn.".format(redundant_token_count)
        )

    return RedundancyReport(
        total_tokens=len(prompt_tokens),
        unique_tokens=unique_tokens,
        redundancy_ratio=round(redundancy_ratio, 3),
        duplicate_ngrams=duplicates[:20],  # top 20
        recommendations=recommendations,
    )
