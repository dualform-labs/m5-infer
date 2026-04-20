"""Lookahead Decoding — generate multiple tokens per forward pass.

Core principle: instead of generating 1 token per forward pass,
feed K "guessed" tokens as a batch. The model processes all K
in one pass (weights loaded once), giving K predictions.
Accept the longest prefix where predictions match guesses.

For Qwen3.5 hybrid (GatedDeltaNet + Full Attention):
- GDN state: save before batch, restore on partial acceptance
- FA KV cache: adjust offset on partial acceptance
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import mlx.core as mx
from app.core.logging import get_logger

logger = get_logger(__name__)


class NGramPredictor:
    """Predicts next tokens based on n-gram patterns from context."""

    def __init__(self, n: int = 3):
        self._n = n
        self._table: dict[tuple, list[int]] = defaultdict(list)

    def update(self, tokens: list[int]) -> None:
        """Add tokens to the n-gram table."""
        for i in range(len(tokens) - self._n):
            key = tuple(tokens[i : i + self._n])
            next_tok = tokens[i + self._n]
            if not self._table[key] or self._table[key][-1] != next_tok:
                self._table[key].append(next_tok)

    def predict(self, context: list[int], num_predict: int = 4) -> list[int]:
        """Predict next tokens based on n-gram match."""
        if len(context) < self._n:
            return []

        key = tuple(context[-self._n :])
        candidates = self._table.get(key, [])

        if not candidates:
            return []

        # Follow the chain
        predicted = []
        for i in range(num_predict):
            if not candidates:
                break
            tok = candidates[0]  # Most recent occurrence
            predicted.append(tok)
            # Extend the key
            key = tuple(list(key[1:]) + [tok])
            candidates = self._table.get(key, [])

        return predicted


@dataclass
class LookaheadStats:
    total_steps: int = 0
    tokens_generated: int = 0
    tokens_accepted: int = 0  # Extra tokens accepted via lookahead
    draft_attempts: int = 0
    draft_tokens: int = 0
    full_rollbacks: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.tokens_accepted / self.draft_tokens if self.draft_tokens > 0 else 0.0

    @property
    def tokens_per_step(self) -> float:
        return self.tokens_generated / self.total_steps if self.total_steps > 0 else 1.0


def save_gdn_states(cache: list, layers: list) -> list[tuple]:
    """Save GatedDeltaNet recurrent + conv states for rollback."""
    saved = []
    for i, layer in enumerate(layers):
        if layer.is_linear:
            conv = cache[i][0]
            rec = cache[i][1]
            # Force copy by creating new arrays
            saved.append((
                i,
                mx.array(conv) if conv is not None else None,
                mx.array(rec) if rec is not None else None,
            ))
    return saved


def restore_gdn_states(cache: list, saved: list[tuple]) -> None:
    """Restore GatedDeltaNet states from snapshot."""
    for i, conv, rec in saved:
        if conv is not None:
            cache[i][0] = conv
        if rec is not None:
            cache[i][1] = rec


def rollback_fa_cache(cache: list, layers: list, num_rollback: int) -> None:
    """Roll back Full Attention KV cache by N positions."""
    for i, layer in enumerate(layers):
        if not layer.is_linear:
            cache[i].offset = max(0, cache[i].offset - num_rollback)
