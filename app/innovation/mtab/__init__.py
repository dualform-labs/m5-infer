"""MTAB — Multi-Tier Activation Bank.

Caches hidden_states at multiple layer boundaries (8, 16, 24, 32) for
common system prompt prefixes. When a new request shares a long prefix
with a cached entry, skip computation up to the cached layer boundary
and resume from the cached hidden_states.

This is a generalization of CTRSP:
  CTRSP: cache the FINAL state (after all 32 layers)
  MTAB:  cache INTERMEDIATE states at multiple layer boundaries

Benefit: partial prefix match → partial layer skip → TTFT 2-5×

Memory cost (per cached context):
  1 tier × 16K tokens × 5120 hidden × 2 bytes = 160 MB
  3 tiers × 5 contexts = ~2.4 GB (Standard tier+)
"""

from app.innovation.mtab.tier_cache import TierCache, TierEntry
from app.innovation.mtab.prefix_matcher import find_longest_prefix_match

__all__ = ["TierCache", "TierEntry", "find_longest_prefix_match"]
