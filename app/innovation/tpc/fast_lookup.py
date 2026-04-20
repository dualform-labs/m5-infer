"""Raw-bytes content hash → compiled-artifact lookup.

This sits IN FRONT of CTRSP / MTAB and short-circuits expensive operations
(tokenization of 12K chars, MTAB sha256 over full token list, redundancy
scan) when the system content matches byte-for-byte a previously seen prompt.

Intended hit path on WARM call:
  raw_sha256(system_content)  -> TPCEntry
  inject entry.token_ids (skip tokenizer)  -> continue with CTRSP using
                                              precomputed hash and prefix_len
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TPCEntry:
    """Cached compilation of a system-role content string.

    Stored fields are CHEAP to compute — expensive KV-cache state lives in
    CTRSP under `token_hash` which is recorded here so the executor can
    bypass compute_prompt_hash().
    """
    content_sha256: str              # 32-hex chars of UTF-8 bytes sha256
    token_ids: tuple[int, ...]       # Pre-tokenized system content
    token_count: int                 # len(token_ids) for quick checks
    ctrsp_token_hash: Optional[str] = None   # SHA-1 over token_ids (CTRSP key)
    mtab_observed: bool = False      # True once background worker has fed MTAB
    redundancy_scanned: bool = False # True once background worker ran redundancy analysis
    hits: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)


class TPCCache:
    """LRU cache keyed by content_sha256 (UTF-8 bytes hash)."""

    def __init__(self, max_entries: int = 128):
        self._entries: OrderedDict[str, TPCEntry] = OrderedDict()
        self._max = max_entries
        self._lock = threading.Lock()
        self._stats = {"lookups": 0, "hits": 0, "misses": 0, "stores": 0, "evictions": 0}

    @staticmethod
    def content_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]

    def lookup(self, content: str) -> Optional[TPCEntry]:
        key = self.content_hash(content)
        with self._lock:
            self._stats["lookups"] += 1
            entry = self._entries.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None
            entry.hits += 1
            entry.last_accessed_at = time.time()
            self._entries.move_to_end(key)
            self._stats["hits"] += 1
            return entry

    def store(self, content: str, token_ids: list[int], ctrsp_token_hash: Optional[str] = None) -> TPCEntry:
        key = self.content_hash(content)
        with self._lock:
            if key in self._entries:
                entry = self._entries[key]
                entry.last_accessed_at = time.time()
                if ctrsp_token_hash and not entry.ctrsp_token_hash:
                    entry.ctrsp_token_hash = ctrsp_token_hash
                self._entries.move_to_end(key)
                return entry

            while len(self._entries) >= self._max:
                self._entries.popitem(last=False)
                self._stats["evictions"] += 1

            entry = TPCEntry(
                content_sha256=key,
                token_ids=tuple(token_ids),
                token_count=len(token_ids),
                ctrsp_token_hash=ctrsp_token_hash,
            )
            self._entries[key] = entry
            self._stats["stores"] += 1
            return entry

    def mark_mtab_observed(self, content: str) -> None:
        key = self.content_hash(content)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry.mtab_observed = True

    def mark_redundancy_scanned(self, content: str) -> None:
        key = self.content_hash(content)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry.redundancy_scanned = True

    def stats(self) -> dict:
        with self._lock:
            total = self._stats["lookups"]
            hit_rate = (self._stats["hits"] / total) if total > 0 else 0.0
            return {
                **self._stats,
                "size": len(self._entries),
                "max": self._max,
                "hit_rate": hit_rate,
            }


_global_cache: Optional[TPCCache] = None
_lock = threading.Lock()


def get_tpc_cache() -> TPCCache:
    """Return process-global TPCCache, creating it if needed."""
    global _global_cache
    if _global_cache is None:
        with _lock:
            if _global_cache is None:
                from app.core.config import get_settings
                settings = get_settings()
                # Reuse prefix_cache_max_entries as upper bound
                max_entries = getattr(settings.cache, "prefix_cache_max_entries", 32) * 4
                _global_cache = TPCCache(max_entries=max_entries)
    return _global_cache
