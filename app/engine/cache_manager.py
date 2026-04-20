"""Basic cache manager (will be extended in Phase 3 with agent-aware features)."""

from __future__ import annotations

import time

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class CacheEntry:
    """A cached item with metadata."""

    def __init__(self, key: str, data, namespace: str = "default"):
        self.key = key
        self.data = data
        self.namespace = namespace
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0

    def touch(self):
        self.last_accessed = time.time()
        self.access_count += 1


class CacheManager:
    """Manages prefix cache, session KV cache, and namespace separation.

    Namespaces: main_prefix, main_session, sub_prefix, sub_session
    """

    def __init__(self):
        self._caches: dict[str, dict[str, CacheEntry]] = {
            "main_prefix": {},
            "main_session": {},
            "sub_prefix": {},
            "sub_session": {},
        }

    def get(self, key: str, namespace: str = "main_prefix") -> CacheEntry | None:
        """Get a cache entry by key and namespace."""
        ns = self._caches.get(namespace, {})
        entry = ns.get(key)
        if entry:
            entry.touch()
            return entry
        return None

    def put(self, key: str, data, namespace: str = "main_prefix") -> None:
        """Store a cache entry."""
        settings = get_settings()
        ns = self._caches.setdefault(namespace, {})

        # Evict if over limit
        max_entries = settings.cache.prefix_cache_max_entries
        if "session" in namespace:
            max_entries = settings.cache.session_cache_max_entries

        while len(ns) >= max_entries:
            self._evict_lru(namespace)

        ns[key] = CacheEntry(key=key, data=data, namespace=namespace)

    def invalidate(self, key: str, namespace: str = "main_prefix") -> bool:
        """Remove a specific cache entry."""
        ns = self._caches.get(namespace, {})
        if key in ns:
            del ns[key]
            return True
        return False

    def clear_namespace(self, namespace: str) -> int:
        """Clear all entries in a namespace."""
        ns = self._caches.get(namespace, {})
        count = len(ns)
        ns.clear()
        return count

    def clear_all(self) -> int:
        """Clear all caches."""
        total = sum(len(ns) for ns in self._caches.values())
        for ns in self._caches.values():
            ns.clear()
        return total

    def stats(self) -> dict[str, int]:
        """Return cache statistics."""
        return {ns: len(entries) for ns, entries in self._caches.items()}

    def get_or_compute_prefix_cache(
        self,
        prompt_hash: str,
        session_id: str | None = None,
    ) -> CacheEntry | None:
        """Get prefix cache entry, checking session-specific first then shared."""
        # Check session-specific cache first
        if session_id:
            entry = self.get(f"{session_id}:{prompt_hash}", "main_session")
            if entry:
                return entry
        # Fall back to shared prefix cache
        return self.get(prompt_hash, "main_prefix")

    def store_prefix_cache(
        self,
        prompt_hash: str,
        data,
        session_id: str | None = None,
        is_system_prompt: bool = False,
    ) -> None:
        """Store a prefix cache entry with priority metadata."""
        namespace = "main_prefix"
        key = prompt_hash
        if session_id:
            namespace = "main_session"
            key = f"{session_id}:{prompt_hash}"
        self.put(key, data, namespace)
        # Mark system prompt entries as high priority (won't be evicted first)
        if is_system_prompt:
            entry = self.get(key, namespace)
            if entry:
                entry.access_count = 1000  # High priority baseline

    def _evict_lru(self, namespace: str) -> None:
        """Evict the least recently used entry from a namespace."""
        ns = self._caches.get(namespace, {})
        if not ns:
            return
        lru_key = min(ns, key=lambda k: ns[k].last_accessed)
        del ns[lru_key]
        logger.info("Evicted cache entry: %s/%s", namespace, lru_key)
