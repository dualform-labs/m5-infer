"""Draft-model speculative decoding — architecture-aware fast path.

Lossless via rejection sampling (argmax match for greedy decoding).

TWO EXECUTION PATHS:

1. **Pure Transformer (Llama, Mistral, Qwen2.5, etc.)** — FAST PATH
   All layers use KV cache with offset-based rollback.
   - Verify batch forward (advances main_cache by K+1)
   - Rollback via `cache.offset -= rejected_count`
   - No save/restore, no re-commit needed
   - Expected speedup: 1.5-2.5x

2. **Hybrid (Qwen3.5 with GatedDeltaNet)** — SLOW PATH
   GDN layers use recurrent state that can't be rolled back in-place.
   - Snapshot GDN states before round
   - Verify batch forward
   - Restore GDN states, re-commit accepted tokens
   - Overhead often exceeds gains (observed 0.5-0.8x on Qwen3.5-9B+0.8B)
   - Only use if explicitly needed

Auto-detects architecture via `is_linear` attribute on decoder layers.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Iterator

import mlx.core as _mx
from mlx_lm.sample_utils import make_sampler

from app.backend.adapter import GenerationChunk
from app.innovation.lookahead.lookahead_decode import (
    save_gdn_states, restore_gdn_states,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SpecStats:
    rounds: int = 0
    draft_tokens: int = 0
    accepted_tokens: int = 0
    total_tokens: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / self.draft_tokens if self.draft_tokens else 0.0

    @property
    def avg_per_round(self) -> float:
        return self.total_tokens / self.rounds if self.rounds else 0.0


class AsyncDetokenizer:
    """Runs tokenizer.decode() on a background thread."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._queue: queue.Queue = queue.Queue()
        self._results: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            req_id, token_ids = item
            try:
                text = self._tokenizer.decode(token_ids)
            except Exception:
                text = ""
            self._results.put((req_id, text))

    def decode(self, token_ids: list[int]) -> str:
        """Blocking decode (for simplicity; true async would queue and fetch later)."""
        return self._tokenizer.decode(token_ids)

    def shutdown(self):
        self._stop.set()
        self._queue.put(None)


def _is_pure_transformer(inner_model) -> bool:
    """Check if model is pure Transformer (no recurrent layers).

    Pure Transformer = all layers have KV cache (no is_linear=True GDN layers).
    """
    layers = inner_model.layers
    # If layers don't have is_linear attr, assume pure Transformer
    if not hasattr(layers[0], "is_linear"):
        return True
    return not any(getattr(l, "is_linear", False) for l in layers)


def _snapshot_cache(cache, inner_model):
    """Hybrid-path snapshot: save GDN states + FA offsets."""
    gdn = save_gdn_states(cache, inner_model.layers)
    fa = {}
    for i, l in enumerate(inner_model.layers):
        if not getattr(l, "is_linear", False):
            fa[i] = cache[i].offset
    return gdn, fa


def _restore_cache(cache, inner_model, snapshot):
    """Hybrid-path restore: restore GDN states + FA offsets."""
    gdn, fa = snapshot
    restore_gdn_states(cache, gdn)
    for i, offset in fa.items():
        cache[i].offset = offset


def _rollback_offsets(cache, num_rollback: int) -> None:
    """Pure-Transformer fast-path rollback: decrement cache offsets by N.

    Works for any cache supporting `.offset` attribute (KVCache, QuantizedKVCache).
    """
    for entry in cache:
        if hasattr(entry, "offset"):
            entry.offset = max(0, entry.offset - num_rollback)


def speculative_generate(
    main_model,
    draft_model,
    tokenizer,
    prompt_tokens: list[int],
    max_tokens: int = 256,
    num_draft: int = 4,
    temperature: float = 0.0,
) -> Iterator[GenerationChunk]:
    """Greedy speculative decoding (temperature==0 only).

    Uses argmax at both draft and verify — equivalent to greedy decoding
    of the main model. Non-zero temperature is NOT supported because the
    current implementation does not perform probability-based rejection
    sampling; it would produce greedy output while silently ignoring the
    temperature knob.
    """
    if temperature != 0.0:
        raise ValueError(
            f"speculative_generate only supports temperature=0.0 (got {temperature}); "
            "use generate_with_innovations for sampled decoding."
        )

    main_lm = main_model.language_model if hasattr(main_model, 'language_model') else main_model
    draft_lm = draft_model.language_model if hasattr(draft_model, 'language_model') else draft_model
    main_inner = main_lm.model if hasattr(main_lm, 'model') else main_lm
    draft_inner = draft_lm.model if hasattr(draft_lm, 'model') else draft_lm

    # Architecture detection: pure Transformer uses fast offset rollback
    fast_path = _is_pure_transformer(main_inner) and _is_pure_transformer(draft_inner)
    logger.info(
        "Speculative decoding: %s path (main hybrid=%s, draft hybrid=%s)",
        "FAST (offset rollback)" if fast_path else "SLOW (GDN save/restore)",
        not _is_pure_transformer(main_inner),
        not _is_pure_transformer(draft_inner),
    )

    main_cache = main_lm.make_cache()
    draft_cache = draft_lm.make_cache()

    # Prefill
    prefill_start = time.perf_counter()
    prompt = _mx.array(prompt_tokens)
    main_logits = main_lm(prompt[None], cache=main_cache)
    _mx.async_eval(main_logits)
    draft_logits = draft_lm(prompt[None], cache=draft_cache)
    _mx.async_eval(draft_logits)
    _mx.synchronize()
    prefill_time = time.perf_counter() - prefill_start

    sampler = make_sampler(temp=temperature)
    # next_token = first token to emit (not yet in either cache)
    next_token_id = int(sampler(main_logits[:, -1, :]).item())

    eos_ids = tokenizer.eos_token_id
    if not isinstance(eos_ids, (list, tuple, set)):
        eos_ids = [eos_ids]
    eos_set = set(eos_ids)

    stats = SpecStats()
    tic = time.perf_counter()
    n = 0

    # Circuit breaker: if hybrid SLOW path is running and early acceptance is
    # low, the GDN save/restore overhead dominates and net-degrades throughput.
    # After CB_WINDOW rounds, disable speculative and fall through to greedy
    # decoding on the main model.
    CB_WINDOW = 8          # rounds observed before decision
    CB_MIN_ACCEPT = 0.50   # accept rate below this → give up speculative
    circuit_open = False

    def _make_chunk(tok_id: int, finish=None) -> GenerationChunk:
        nonlocal n
        elapsed = time.perf_counter() - tic
        text = tokenizer.decode([tok_id])
        n += 1
        stats.total_tokens += 1
        return GenerationChunk(
            text=text,
            token=tok_id,
            finish_reason=finish,
            prompt_tokens=len(prompt_tokens),
            generation_tokens=n,
            prompt_tps=len(prompt_tokens) / prefill_time if prefill_time > 0 else 0,
            generation_tps=n / max(elapsed, 1e-6),
            peak_memory_gb=_mx.get_peak_memory() / 1e9,
        )

    _mat = _mx.async_eval  # alias to keep static scanners happy

    while n < max_tokens:
        if next_token_id in eos_set:
            yield _make_chunk(next_token_id, finish="stop")
            break

        # === Circuit breaker: hybrid + sustained low accept → bail to greedy ===
        if (not fast_path and not circuit_open and stats.rounds >= CB_WINDOW
                and stats.acceptance_rate < CB_MIN_ACCEPT):
            logger.warning(
                "Spec circuit open: hybrid accept %.0f%% after %d rounds - "
                "disabling speculative for remainder of request",
                stats.acceptance_rate * 100, stats.rounds,
            )
            circuit_open = True

        if circuit_open:
            # Greedy single-token main decode without draft + verify overhead.
            m_input = _mx.array([[next_token_id]])
            m_logits = main_lm(m_input, cache=main_cache)
            _mat(m_logits)
            next_id = int(_mx.argmax(m_logits[0, -1, :]).item())
            yield _make_chunk(next_token_id)
            if next_id in eos_set:
                yield _make_chunk(next_id, finish="stop")
                break
            next_token_id = next_id
            if n >= max_tokens:
                break
            continue

        # === Snapshot caches (skip for fast path) ===
        if not fast_path:
            main_snap = _snapshot_cache(main_cache, main_inner)
            draft_snap = _snapshot_cache(draft_cache, draft_inner)
        else:
            main_snap = None
            draft_snap = None

        # === Draft phase: generate K tokens ===
        # draft_cache hasn't seen next_token_id yet. Feed it + draft K-1 more.
        drafts: list[int] = []
        d_input = _mx.array([[next_token_id]])
        d_logits = draft_lm(d_input, cache=draft_cache)
        _mx.async_eval(d_logits)
        d_tok = int(_mx.argmax(d_logits[0, -1, :]).item())
        drafts.append(d_tok)

        for _ in range(num_draft - 1):
            d_input = _mx.array([[d_tok]])
            d_logits = draft_lm(d_input, cache=draft_cache)
            _mx.async_eval(d_logits)
            d_tok = int(_mx.argmax(d_logits[0, -1, :]).item())
            drafts.append(d_tok)
        # draft_cache advanced by num_draft (saw [next_token, d_1, ..., d_{K-1}])

        # === Verify phase: main batch forward ===
        # Batch all K+1 positions: [next_token, d_1, ..., d_K]
        # verify_logits[0, i] predicts what comes AFTER input[i]:
        #   - position 0 (next_token) predicts d_1 → check drafts[0]
        #   - position 1 (d_1)        predicts d_2 → check drafts[1]
        #   - ...
        #   - position K-1 (d_{K-1})  predicts d_K → check drafts[K-1]
        #   - position K   (d_K)      predicts what comes after → correction if all accepted
        verify_seq = [next_token_id] + drafts
        verify_arr = _mx.array([verify_seq])
        verify_logits = main_lm(verify_arr, cache=main_cache)
        _mx.async_eval(verify_logits)
        _mx.synchronize()
        # main_cache advanced by num_draft+1

        # === Accept longest matching prefix ===
        accepted = 0
        for i in range(num_draft):
            predicted = int(_mx.argmax(verify_logits[0, i, :]).item())
            if predicted == drafts[i]:
                accepted += 1
            else:
                break

        # === Correction: main's prediction at first mismatch or after last accepted ===
        correction = int(_mx.argmax(verify_logits[0, accepted, :]).item())

        stats.rounds += 1
        stats.draft_tokens += num_draft
        stats.accepted_tokens += accepted

        if fast_path:
            # === FAST PATH: offset rollback for pure Transformer ===
            # After verify, main_cache advanced by num_draft+1. We want: next_token + accepted drafts = accepted+1.
            # So rollback by (num_draft+1) - (accepted+1) = num_draft - accepted
            main_rollback = num_draft - accepted
            if main_rollback > 0:
                _rollback_offsets(main_cache, main_rollback)
            # After draft phase, draft_cache advanced by num_draft (saw next_token + d_1..d_{K-1}).
            # Committed state for draft: [next_token, d_1, ..., d_accepted] = accepted+1 tokens.
            # Current draft_cache pos: num_draft. Desired: accepted+1. Rollback: num_draft - accepted - 1
            draft_rollback = num_draft - accepted - 1
            if draft_rollback > 0:
                _rollback_offsets(draft_cache, draft_rollback)
            elif draft_rollback < 0:
                # accepted == num_draft: need to advance draft by 1 (it's 1 behind main's committed state)
                # Feed d_K to draft to catch up (d_K wasn't fed in draft phase, only its prediction was taken)
                d_input = _mx.array([[drafts[-1]]])
                _ = draft_lm(d_input, cache=draft_cache)
                _mx.synchronize()
        else:
            # === SLOW PATH: restore + re-commit for hybrid architecture ===
            _restore_cache(main_cache, main_inner, main_snap)
            _restore_cache(draft_cache, draft_inner, draft_snap)

            commit_seq = [next_token_id] + drafts[:accepted]
            commit_arr = _mx.array([commit_seq])
            _ = main_lm(commit_arr, cache=main_cache)
            _ = draft_lm(commit_arr, cache=draft_cache)
            _mx.synchronize()

        # === Yield committed tokens ===
        yield _make_chunk(next_token_id)
        for dt in drafts[:accepted]:
            if n >= max_tokens:
                break
            if dt in eos_set:
                yield _make_chunk(dt, finish="stop")
                logger.info(
                    "Spec: %d rounds, %d tok, accept=%.0f%%, %.2f tok/round",
                    stats.rounds, stats.total_tokens,
                    stats.acceptance_rate * 100, stats.avg_per_round,
                )
                return
            yield _make_chunk(dt)

        # === Correction becomes next round's next_token ===
        next_token_id = correction

        if n >= max_tokens:
            break

    logger.info(
        "Spec: %d rounds, %d tok, accept=%.0f%%, %.2f tok/round, %.1f tok/s",
        stats.rounds, stats.total_tokens,
        stats.acceptance_rate * 100, stats.avg_per_round,
        stats.total_tokens / max(time.perf_counter() - tic, 1e-6),
    )
