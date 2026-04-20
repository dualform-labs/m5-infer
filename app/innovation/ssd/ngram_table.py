"""SSD n-gram lookup table.

Stores deterministic n-gram → next-token mappings, e.g.:
  (token_a, token_b, token_c) → (next_token_d, confidence=0.98)

Loaded from disk (per-model JSON) at startup. Empty by default —
the table must be built offline via `ngram_builder.py` for each
tokenizer/model pair.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)


class SSDTable:
    """Lookup table for deterministic n-gram patterns."""

    def __init__(self):
        # Maps tuple of recent token ids → (predicted_token_id, confidence)
        self._table: dict[tuple, tuple[int, float]] = {}
        self._lock = threading.RLock()
        self._stats = {
            "lookups": 0,
            "hits": 0,
            "misses": 0,
            "verify_correct": 0,
            "verify_mismatch": 0,
        }

    def load_from_file(self, path: str) -> bool:
        """Load table from JSON file. Format:
            {
              "ngram_size": 5,
              "min_confidence": 0.95,
              "entries": [
                {"context": [101, 102], "next": 103, "confidence": 0.99},
                ...
              ]
            }

        Returns True on success.
        """
        p = Path(path)
        if not p.is_file():
            logger.info("SSD: no table file at %s — SSD will be inert", path)
            return False
        try:
            data = json.loads(p.read_text())
            entries = data.get("entries", [])
            with self._lock:
                self._table.clear()
                for e in entries:
                    ctx = tuple(e["context"])
                    nxt = int(e["next"])
                    conf = float(e.get("confidence", 1.0))
                    self._table[ctx] = (nxt, conf)
            logger.info("SSD: loaded %d entries from %s", len(entries), path)
            return True
        except Exception:
            logger.exception("SSD: failed to load table from %s", path)
            return False

    def lookup(self, context: list[int], min_confidence: float = 0.95) -> int | None:
        """Look up the most-recent N-gram in context.

        Tries longest match first (n down to 2).
        Returns predicted token id if confidence >= min_confidence, else None.
        """
        with self._lock:
            self._stats["lookups"] += 1
            # Try lengths from longest possible (5) down to 2
            for n in range(min(5, len(context)), 1, -1):
                key = tuple(context[-n:])
                if key in self._table:
                    pred, conf = self._table[key]
                    if conf >= min_confidence:
                        self._stats["hits"] += 1
                        return pred
            self._stats["misses"] += 1
            return None

    def record_verify(self, was_correct: bool) -> None:
        """Track verify-mode results for adaptive threshold tuning."""
        with self._lock:
            if was_correct:
                self._stats["verify_correct"] += 1
            else:
                self._stats["verify_mismatch"] += 1

    def size(self) -> int:
        with self._lock:
            return len(self._table)

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats, size=len(self._table))


# Module-level singleton — loaded once
_singleton: SSDTable | None = None


def get_ssd_table() -> SSDTable:
    """Return the singleton SSDTable.

    On first access, attempts to load from `state/ssd_table_<model>.json`
    if available. Otherwise the table starts empty (SSD becomes a no-op).
    """
    global _singleton
    if _singleton is None:
        _singleton = SSDTable()
        # Best-effort load; silent if file doesn't exist
        default_path = Path("state/ssd_table.json")
        if default_path.exists():
            _singleton.load_from_file(str(default_path))
    return _singleton
