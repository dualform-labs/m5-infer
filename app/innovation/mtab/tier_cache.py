"""MTAB Tier Cache — stores hidden_states snapshots at layer boundaries."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


# Layer boundaries where we cache hidden_states.
# For Qwen3.5 (32 layers), these are 1/4, 1/2, 3/4 boundaries.
# Allows "partial restoration" — skip first N layers if prefix matches.
DEFAULT_TIER_BOUNDARIES = (8, 16, 24)


@dataclass
class TierEntry:
    """One cached snapshot at a specific layer boundary."""
    prompt_hash: str            # sha256 of the prompt prefix tokens
    prompt_tokens: list[int]    # the actual tokens (for prefix-match check)
    layer_boundary: int         # which layer's output is cached (e.g., 16)
    hidden_states: Any          # mx.array of shape [1, T, hidden_dim]
    last_used_at: float = 0.0


class TierCache:
    """LRU cache for multi-tier activations.

    Multiple tiers (layer boundaries) per prompt; lookup returns the
    deepest matching tier (= maximum compute saved).
    """

    def __init__(
        self,
        max_entries: int = 16,
        boundaries: tuple[int, ...] = DEFAULT_TIER_BOUNDARIES,
    ):
        self._max_entries = max(1, int(max_entries))
        self._boundaries = boundaries
        # OrderedDict for true LRU
        self._cache: "OrderedDict[str, TierEntry]" = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            "lookups": 0,
            "exact_hits": 0,
            "partial_hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
        }

    def store(self, entry: TierEntry) -> None:
        """Insert a new tier entry."""
        with self._lock:
            key = self._make_key(entry.prompt_hash, entry.layer_boundary)
            if key in self._cache:
                self._cache.pop(key)
            while len(self._cache) >= self._max_entries:
                old_key, _ = self._cache.popitem(last=False)
                self._stats["evictions"] += 1
                logger.debug("MTAB: evicted %s", old_key)
            self._cache[key] = entry
            self._stats["stores"] += 1

    def lookup_best(
        self, prompt_tokens: list[int],
    ) -> tuple[TierEntry | None, int]:
        """Find best matching tier entry for the given prompt.

        Returns (entry, match_length) where:
          - entry is the cached TierEntry or None
          - match_length is the # of tokens that match
        """
        with self._lock:
            self._stats["lookups"] += 1
            best_entry: TierEntry | None = None
            best_match = 0
            best_boundary = 0

            for entry in self._cache.values():
                cached = entry.prompt_tokens
                # How many tokens of `prompt_tokens` match the cached prefix?
                n = min(len(cached), len(prompt_tokens))
                m = 0
                while m < n and prompt_tokens[m] == cached[m]:
                    m += 1
                # Prefer entries with longer match AND deeper layer boundary
                # (deeper = more compute saved per matched token)
                if m == 0:
                    continue
                # Score = (match_length, layer_boundary) lex order
                if (m, entry.layer_boundary) > (best_match, best_boundary):
                    best_entry = entry
                    best_match = m
                    best_boundary = entry.layer_boundary

            if best_entry is None:
                self._stats["misses"] += 1
                return None, 0

            # Move to MRU
            for k, v in list(self._cache.items()):
                if v is best_entry:
                    self._cache.move_to_end(k)
                    break

            if best_match == len(best_entry.prompt_tokens):
                self._stats["exact_hits"] += 1
            else:
                self._stats["partial_hits"] += 1

            return best_entry, best_match

    def boundaries(self) -> tuple[int, ...]:
        return self._boundaries

    def clear(self) -> int:
        with self._lock:
            n = len(self._cache)
            self._cache.clear()
            return n

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats, size=len(self._cache))

    def _make_key(self, prompt_hash: str, layer_boundary: int) -> str:
        return f"{prompt_hash}:{layer_boundary}"


# Module-level singleton (initialized on first access by mode-aware factory)
_singleton: TierCache | None = None


def get_tier_cache() -> TierCache:
    global _singleton
    if _singleton is None:
        from app.core.config import get_settings
        feat = get_settings().runtime.active_features()
        # MTAB cache size scales with mode; use small default for Base
        size = 5 if not feat.mtab_enabled else 16
        _singleton = TierCache(max_entries=size)
        logger.info(
            "MTAB: TierCache initialized (size=%d, enabled=%s)",
            size, feat.mtab_enabled,
        )
    return _singleton
