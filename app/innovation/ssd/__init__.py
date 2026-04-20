"""SSD — Structural Skip Decoding.

Bypass the model entirely for "structurally deterministic" tokens. Common
patterns (markdown headers, JSON brackets, common phrases) have such a
high probability of a specific next token that running the full model is
wasteful — a small lookup table suffices.

Pipeline:
  1. Offline: scan a large corpus, find n-gram → next-token mappings
     where the next token has > 0.95 conditional probability.
  2. Runtime: at each decode step, look up the recent k tokens in the
     table. If a high-confidence match exists, emit that token without
     calling the model.
  3. Verify mode: periodically (every Nth skip), compare against actual
     model prediction. If mismatch, fall back and disable the entry.

Memory cost: ~5 MB JSON table, negligible.
Expected effect: 30-50% of tokens skip model → decode +40% (extreme tier).
"""

from app.innovation.ssd.ngram_table import SSDTable, get_ssd_table
from app.innovation.ssd.skip_predictor import SkipPredictor

__all__ = ["SSDTable", "get_ssd_table", "SkipPredictor"]
