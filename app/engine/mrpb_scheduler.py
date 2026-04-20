"""MRPB — Multi-Request Parallel Batching scheduler.

Splits the monolithic `_generation_lock` into per-request-class locks so that
non-conflicting requests can run concurrently (given MLX multi-stream support).

Request classes
---------------
- "heavy"   — long prompt prefill, >1K output tokens, tool-heavy
- "medium"  — standard chat (≤ 1K output)
- "light"   — ≤ 64 output tokens, short classifier-style calls

On a single-model 24 GB engine MLX streams still share the same KV cache, so
genuine parallelism is not safe. The scheduler therefore:

  * Always serializes within the same class (correctness).
  * Allows cross-class overlap only when `is_parallel_enabled()` is True.
    That flag stays off until the MMRS/Pro tier path verifies MLX multi-stream
    correctness on a per-session KV cache.

This keeps the single-user 24 GB path bit-for-bit identical to Phase B while
providing the hooks the Pro tier will light up.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Literal

from app.core.logging import get_logger

logger = get_logger(__name__)

RequestClass = Literal["heavy", "medium", "light"]


def classify_request(max_tokens: int, prompt_tokens_estimate: int = 0) -> RequestClass:
    if max_tokens <= 64 and prompt_tokens_estimate < 2000:
        return "light"
    if max_tokens >= 1024 or prompt_tokens_estimate >= 8000:
        return "heavy"
    return "medium"


class MRPBScheduler:
    """Per-class asyncio locks + a master lock that serializes by default."""

    def __init__(self) -> None:
        self._class_locks: dict[str, asyncio.Lock] = {
            "heavy": asyncio.Lock(),
            "medium": asyncio.Lock(),
            "light": asyncio.Lock(),
        }
        self._master = asyncio.Lock()
        self._parallel = False  # gated off on 24 GB M5 base

    def enable_parallel(self) -> None:
        """Only call after verifying multi-stream KV isolation (Pro tier)."""
        self._parallel = True
        logger.info("MRPB parallel mode enabled")

    def is_parallel_enabled(self) -> bool:
        return self._parallel

    @asynccontextmanager
    async def acquire(self, cls: RequestClass):
        """Acquire the appropriate lock for the request class."""
        if self._parallel:
            lock = self._class_locks[cls]
            async with lock:
                yield
        else:
            async with self._master:
                yield

    def stats(self) -> dict:
        return {
            "parallel_enabled": self._parallel,
            "master_locked": self._master.locked(),
            "class_locked": {k: v.locked() for k, v in self._class_locks.items()},
        }


_global_scheduler: MRPBScheduler | None = None


def get_mrpb_scheduler() -> MRPBScheduler:
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = MRPBScheduler()
    return _global_scheduler
