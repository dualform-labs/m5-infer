"""N1: Cross-Turn Recurrent State Persistence (CTRSP).

Caches GatedDeltaNet recurrent state and conv state between agent turns,
enabling 5-10x TTFT improvement on subsequent turns by skipping
system prompt re-processing through 24/32 layers.

Mathematical guarantee: f(A+B, S0) = f(B, f(A, S0)) for sequential recurrence.
Validated by Codex review.
"""

from __future__ import annotations
import hashlib
import json
import os
import threading
import time
import mlx.core as mx
import numpy as np
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from app.core.logging import get_logger

logger = get_logger(__name__)

_META_SCHEMA_VERSION = 2


@dataclass
class CachedModelState:
    """Cached state for a specific system prompt + model combination."""
    prompt_hash: str
    model_name: str
    gdn_states: list[tuple[mx.array, mx.array]]
    kv_cache_keys: list
    tokens_processed: int
    position_offset: int

    def memory_bytes(self) -> int:
        total = 0
        for recurrent, conv in self.gdn_states:
            total += recurrent.nbytes + conv.nbytes
        return total


class CTRSPManager:
    """Manages cross-turn recurrent state caching.

    Supports optional disk persistence: states are saved as .npz files
    (NumPy compressed) with JSON metadata. Survives server restarts.

    Thread-safe via a single lock around all mutating operations.
    Writes are atomic (tmp + os.replace) to avoid corruption on crash.
    """

    def __init__(
        self,
        max_cached_states: int = 4,
        persist_dir: str | None = None,
        weights_fingerprint: str | None = None,
    ):
        # OrderedDict tracks true LRU via move_to_end on read.
        self._cache: "OrderedDict[str, CachedModelState]" = OrderedDict()
        self._max_states = max(1, int(max_cached_states))
        self._persist_dir = Path(persist_dir) if persist_dir else None
        self._weights_fingerprint = weights_fingerprint or ""
        self._lock = threading.RLock()
        if self._persist_dir:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    def compute_prompt_hash(self, system_prompt_tokens: list[int]) -> str:
        """Collision-resistant hash over tokens."""
        token_str = ",".join(str(t) for t in system_prompt_tokens)
        return hashlib.sha256(token_str.encode()).hexdigest()[:32]

    def set_weights_fingerprint(self, fingerprint: str) -> None:
        """Set the current model weights fingerprint. Entries whose
        fingerprint differs are invalidated (both memory and disk)."""
        with self._lock:
            if fingerprint == self._weights_fingerprint:
                return
            self._weights_fingerprint = fingerprint
            self._cache.clear()
            if self._persist_dir:
                for p in list(self._persist_dir.iterdir()):
                    try:
                        p.unlink()
                    except OSError:
                        pass
            logger.info("CTRSP: weights fingerprint changed, cache cleared")

    # ------------------------------------------------------------------
    # Cache ops
    # ------------------------------------------------------------------

    def _key(self, model_name: str, prompt_hash: str) -> str:
        return f"{model_name}:{prompt_hash}"

    def get_cached_state(self, prompt_hash: str, model_name: str) -> CachedModelState | None:
        key = self._key(model_name, prompt_hash)
        with self._lock:
            state = self._cache.get(key)
            if state is not None:
                # Move to end = most recently used
                self._cache.move_to_end(key)
                logger.info(
                    "CTRSP hit: %s (%d tokens, %d GDN layers)",
                    key, state.tokens_processed, len(state.gdn_states),
                )
            return state

    def save_state(
        self,
        prompt_hash: str,
        model_name: str,
        gdn_states: list[tuple[mx.array, mx.array]],
        kv_cache_keys: list,
        tokens_processed: int,
        position_offset: int,
    ) -> None:
        key = self._key(model_name, prompt_hash)

        # Upcast once, outside the lock (MLX ops)
        saved_gdn = []
        for recurrent, conv in gdn_states:
            saved_gdn.append((
                recurrent.astype(mx.float32) if recurrent.dtype != mx.float32 else recurrent,
                conv.astype(mx.float32) if conv.dtype != mx.float32 else conv,
            ))
        # Force materialization of lazy arrays
        mx.eval(*[s for pair in saved_gdn for s in pair])

        state = CachedModelState(
            prompt_hash=prompt_hash,
            model_name=model_name,
            gdn_states=saved_gdn,
            kv_cache_keys=kv_cache_keys,
            tokens_processed=tokens_processed,
            position_offset=position_offset,
        )

        with self._lock:
            # If updating an existing key, remove it first so eviction is
            # evaluated on the post-insertion count.
            if key in self._cache:
                self._cache.pop(key)
            # Evict LRU (front of the ordered dict) until we fit.
            while len(self._cache) >= self._max_states:
                oldest_key, oldest_state = self._cache.popitem(last=False)
                self._delete_disk(oldest_key)
                logger.info(
                    "CTRSP evicted: %s (%.2f MB)",
                    oldest_key, oldest_state.memory_bytes() / 1e6,
                )
            self._cache[key] = state
            logger.info(
                "CTRSP saved: %s (%d tokens, %d layers, %.2f MB)",
                key, tokens_processed, len(saved_gdn), state.memory_bytes() / 1e6,
            )
            if self._persist_dir:
                self._persist_entry(key, state)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _safe_name(self, key: str) -> str:
        return key.replace(":", "_").replace("/", "_")

    def _paths(self, key: str) -> tuple[Path, Path]:
        safe = self._safe_name(key)
        return (self._persist_dir / f"{safe}.npz", self._persist_dir / f"{safe}.json")

    def _delete_disk(self, key: str) -> None:
        if not self._persist_dir:
            return
        npz, js = self._paths(key)
        for p in (npz, js):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
            except OSError as e:
                logger.warning("CTRSP delete failed: %s (%s)", p, e)

    def _persist_entry(self, key: str, state: CachedModelState) -> None:
        """Atomic write: write to a distinct tmp path then os.replace to final.

        Note: np.savez_compressed appends `.npz` to the given path if it does
        not already end in `.npz`, so we write to a sibling `.partial.npz`
        and rename to the final `.npz`.
        """
        npz_path, json_path = self._paths(key)
        npz_tmp = npz_path.with_name(npz_path.stem + ".partial.npz")
        json_tmp = json_path.with_name(json_path.stem + ".partial.json")
        try:
            arrays: dict = {}
            for i, (rec, conv) in enumerate(state.gdn_states):
                arrays[f"rec_{i}"] = np.array(rec)
                arrays[f"conv_{i}"] = np.array(conv)
            np.savez_compressed(npz_tmp, **arrays)

            meta = {
                "schema_version": _META_SCHEMA_VERSION,
                "weights_fingerprint": self._weights_fingerprint,
                "prompt_hash": state.prompt_hash,
                "model_name": state.model_name,
                "tokens_processed": state.tokens_processed,
                "position_offset": state.position_offset,
                "num_gdn_states": len(state.gdn_states),
                "last_used": time.time(),
            }
            with open(json_tmp, "w") as f:
                json.dump(meta, f)

            os.replace(npz_tmp, npz_path)
            os.replace(json_tmp, json_path)
            logger.debug("CTRSP persisted: %s", npz_path.name)
        except Exception as e:
            logger.warning("CTRSP persist failed: %s", e)
            for p in (npz_tmp, json_tmp):
                try:
                    p.unlink()
                except Exception:
                    pass

    def _load_from_disk(self) -> None:
        """Load persisted entries, validate fingerprint, trim overflow."""
        candidates: list[tuple[float, Path, Path, dict]] = []
        for json_path in self._persist_dir.glob("*.json"):
            npz_path = json_path.with_suffix(".npz")
            if not npz_path.exists():
                # orphan metadata — clean up
                try:
                    json_path.unlink()
                except OSError:
                    pass
                continue
            try:
                with open(json_path) as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("CTRSP corrupt meta %s: %s — removing", json_path.name, e)
                self._delete_pair(npz_path, json_path)
                continue

            # Schema / fingerprint check
            if meta.get("schema_version") != _META_SCHEMA_VERSION:
                logger.info("CTRSP schema mismatch %s — removing", json_path.name)
                self._delete_pair(npz_path, json_path)
                continue
            if self._weights_fingerprint and meta.get("weights_fingerprint") != self._weights_fingerprint:
                logger.info("CTRSP fingerprint mismatch %s — removing", json_path.name)
                self._delete_pair(npz_path, json_path)
                continue

            candidates.append((meta.get("last_used", 0.0), npz_path, json_path, meta))

        # Keep most-recently-used top N; discard the rest from disk.
        candidates.sort(key=lambda x: x[0], reverse=True)
        keep = candidates[: self._max_states]
        for _, npz, js, _ in candidates[self._max_states:]:
            self._delete_pair(npz, js)

        loaded = 0
        for last_used, npz_path, json_path, meta in keep:
            try:
                npz = np.load(npz_path)
                n_states = int(meta["num_gdn_states"])
                gdn_states = []
                for i in range(n_states):
                    rec = mx.array(npz[f"rec_{i}"])
                    conv = mx.array(npz[f"conv_{i}"])
                    gdn_states.append((rec, conv))

                state = CachedModelState(
                    prompt_hash=meta["prompt_hash"],
                    model_name=meta["model_name"],
                    gdn_states=gdn_states,
                    kv_cache_keys=[],  # FA KV is not disk-persisted
                    tokens_processed=meta["tokens_processed"],
                    position_offset=meta["position_offset"],
                )
                key = self._key(state.model_name, state.prompt_hash)
                self._cache[key] = state
                loaded += 1
            except Exception as e:
                logger.warning("CTRSP load failed %s: %s — removing", json_path.name, e)
                self._delete_pair(npz_path, json_path)

        if loaded > 0:
            logger.info("CTRSP loaded %d states from disk", loaded)

    def _delete_pair(self, npz: Path, js: Path) -> None:
        for p in (npz, js):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Restore
    # ------------------------------------------------------------------

    def restore_gdn_states(
        self,
        cached: CachedModelState,
        target_dtype: mx.Dtype = mx.bfloat16,
    ) -> list[tuple[mx.array, mx.array]]:
        restored = []
        for recurrent, conv in cached.gdn_states:
            restored.append((
                recurrent.astype(target_dtype),
                conv.astype(target_dtype),
            ))
        return restored

    def clear(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            if self._persist_dir:
                for p in list(self._persist_dir.iterdir()):
                    try:
                        p.unlink()
                    except OSError:
                        pass
            return count

    def stats(self) -> dict:
        with self._lock:
            total_bytes = sum(s.memory_bytes() for s in self._cache.values())
            return {
                "cached_states": len(self._cache),
                "total_memory_mb": round(total_bytes / 1e6, 2),
                "entries": [
                    {"key": k, "tokens": v.tokens_processed, "layers": len(v.gdn_states)}
                    for k, v in self._cache.items()
                ],
            }
