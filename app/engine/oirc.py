"""T14-OIRC — Opt-In Response Cache (agent-safe).

This module replaces the original T14 DRC (Deterministic Response Cache).
DRC was removed on 2026-04-22 because it broke a critical agent-workflow
assumption: re-issuing the same LLM request is often intentional ("re-check
current state"), and returning a cached response silently serves stale
semantics.

OIRC fixes this by flipping the default: caching is **never** automatic.
The caller must **explicitly opt in per request** by supplying an
``idempotency_key`` and a positive ``cache_ttl_ms`` (bounded by server
policy). A request without both fields takes the normal generation path
every time — exactly what an agent calling "re-check" expects.

The design mirrors the HTTP industry pattern (Stripe, Plaid, AWS):
idempotency keys are supplied by the caller, scoped to their business
semantics, and only honored within a short TTL. Keys are never derived
from the prompt bytes alone; the caller must make that choice.

Key design points
-----------------
1. **Default off.** A request with no ``idempotency_key`` or with
   ``cache_ttl_ms <= 0`` is always executed fresh. Agent re-check works.
2. **TTL is strictly bounded.** The server clamps ``cache_ttl_ms`` to
   ``OIRC_MAX_TTL_MS`` (60 000 ms) regardless of the caller's request.
   Long-lived caches are simply not allowed.
3. **Key scope includes everything that determines output.** The cache
   key is ``(model_id, idempotency_key, prompt_hash, max_tokens,
   enable_thinking)`` so accidental collisions between callers or
   prompt variants cannot return the wrong response.
4. **Tools disable OIRC.** When ``tools`` are present on the request,
   OIRC is bypassed unconditionally — tool-calling has side effects by
   definition, and caching its output would violate the side-effect
   contract on replay.
5. **LRU + TTL.** Entries expire on read if past their deadline; an
   LRU bound (``OIRC_MAX_ENTRIES``) limits memory growth.

Safety contract
---------------
- Cache replay is byte-identical to the original response because the key
  covers all inputs that determine the output, AND the caller has
  affirmatively asserted "I accept a cached response for this key".
- An agent that wants a fresh run simply omits the ``idempotency_key`` (or
  omits ``cache_ttl_ms``). No new agent-facing code is required to
  preserve re-check semantics.
- Stochastic requests (``temperature > 0`` / ``top_p < 1``) are
  short-circuited before the cache is even consulted.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock


# Tunables (module-level constants so server policy is visible at a glance).
OIRC_MAX_ENTRIES = 128
OIRC_MAX_TTL_MS = 60_000           # 1 minute hard cap
OIRC_MAX_TEXT_CHARS = 16_384       # ~4 KB per entry upper bound
OIRC_MAX_TOKENS_CACHED = 2_048     # do not cache huge generations


@dataclass
class CachedChunk:
    text: str
    finish_reason: str | None = None
    generation_tokens: int = 0


@dataclass
class CachedResponse:
    chunks: list[CachedChunk] = field(default_factory=list)
    total_text_len: int = 0
    expires_at: float = 0.0        # monotonic clock deadline
    completed: bool = False


class OIRCCache:
    def __init__(self, max_entries: int = OIRC_MAX_ENTRIES):
        self._cache: "OrderedDict[str, CachedResponse]" = OrderedDict()
        self._max_entries = max_entries
        self._lock = Lock()
        self._stats = {"hits": 0, "misses": 0, "stores": 0, "rejects": 0, "expired": 0}

    @staticmethod
    def _key(
        model_id: str,
        idempotency_key: str,
        prompt_tokens: list[int],
        max_tokens: int,
        enable_thinking: bool,
    ) -> str:
        h = hashlib.sha256()
        h.update(model_id.encode("utf-8", errors="replace"))
        h.update(b"\x00ik=")
        h.update(idempotency_key.encode("utf-8", errors="replace"))
        h.update(b"\x01mt=")
        h.update(str(int(max_tokens)).encode())
        h.update(b"\x02et=")
        h.update(b"1" if enable_thinking else b"0")
        h.update(b"\x03p=")
        buf = bytearray(len(prompt_tokens) * 4)
        for i, t in enumerate(prompt_tokens):
            t_i = int(t) & 0xFFFFFFFF
            buf[i * 4]     = (t_i >> 24) & 0xFF
            buf[i * 4 + 1] = (t_i >> 16) & 0xFF
            buf[i * 4 + 2] = (t_i >> 8) & 0xFF
            buf[i * 4 + 3] = t_i & 0xFF
        h.update(bytes(buf))
        return h.hexdigest()

    def get(
        self,
        model_id: str,
        idempotency_key: str,
        prompt_tokens: list[int],
        max_tokens: int,
        enable_thinking: bool,
    ) -> CachedResponse | None:
        key = self._key(model_id, idempotency_key, prompt_tokens, max_tokens, enable_thinking)
        now = time.monotonic()
        with self._lock:
            resp = self._cache.get(key)
            if resp is None:
                self._stats["misses"] += 1
                return None
            if not resp.completed:
                self._stats["misses"] += 1
                return None
            if resp.expires_at <= now:
                # Expired — drop the entry so future callers see a true miss.
                self._cache.pop(key, None)
                self._stats["expired"] += 1
                self._stats["misses"] += 1
                return None
            # Fresh hit — promote in LRU.
            self._cache.move_to_end(key)
            self._stats["hits"] += 1
            return resp

    def put(
        self,
        model_id: str,
        idempotency_key: str,
        prompt_tokens: list[int],
        max_tokens: int,
        enable_thinking: bool,
        chunks: list[CachedChunk],
        ttl_ms: int,
    ) -> bool:
        """Persist a completed response for the given idempotency key.

        Returns True if stored, False if rejected (oversized / zero TTL / etc.).
        """
        ttl_ms = min(int(ttl_ms), OIRC_MAX_TTL_MS)
        if ttl_ms <= 0:
            self._stats["rejects"] += 1
            return False
        total_len = sum(len(c.text) for c in chunks)
        total_toks = sum(c.generation_tokens for c in chunks) if chunks else 0
        if total_len > OIRC_MAX_TEXT_CHARS:
            self._stats["rejects"] += 1
            return False
        if total_toks > OIRC_MAX_TOKENS_CACHED:
            self._stats["rejects"] += 1
            return False
        key = self._key(model_id, idempotency_key, prompt_tokens, max_tokens, enable_thinking)
        expires_at = time.monotonic() + ttl_ms / 1000.0
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
            while len(self._cache) >= self._max_entries:
                self._cache.popitem(last=False)
            self._cache[key] = CachedResponse(
                chunks=list(chunks),
                total_text_len=total_len,
                expires_at=expires_at,
                completed=True,
            )
            self._stats["stores"] += 1
            return True

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)


# Process-wide singleton.
_oirc_singleton: OIRCCache | None = None


def get_oirc() -> OIRCCache:
    global _oirc_singleton
    if _oirc_singleton is None:
        _oirc_singleton = OIRCCache()
    return _oirc_singleton


def is_oirc_eligible(
    idempotency_key: str | None,
    cache_ttl_ms: int | None,
    temperature: float,
    top_p: float,
    max_tokens: int | None,
    tools: list | None,
) -> bool:
    """Strict eligibility predicate — every condition must be met.

    The asymmetry is intentional: without all of these, we do not even
    consult the cache. Agent re-check is the default; caching is the
    opt-in exception.
    """
    if not idempotency_key:                              # explicit caller intent
        return False
    if cache_ttl_ms is None or cache_ttl_ms <= 0:        # explicit TTL
        return False
    if tools:                                            # tool calls are side-effectful
        return False
    if temperature != 0.0:                               # stochastic
        return False
    if top_p is not None and top_p < 1.0:                # stochastic
        return False
    if max_tokens is None or max_tokens <= 0:
        return False
    if max_tokens > OIRC_MAX_TOKENS_CACHED:
        return False
    return True
