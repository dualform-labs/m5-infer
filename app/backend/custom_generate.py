"""Custom generation loop with Innovation hooks.

Replaces mlx_lm's generate_step/stream_generate with a version that
integrates Innovation techniques at the model-layer level:

- N1 CTRSP: Inject/extract GatedDeltaNet state from cache before/after generation
- N4 ALS: Skip low-impact layers during decode (custom forward pass)
- X2 DPC: (future) Swap layer weights to 2-bit for high-confidence tokens

The key innovation: custom_forward() replaces the model's __call__ in the
decode loop, allowing per-layer hooks (skip, profile, etc).
"""

from __future__ import annotations

import time
from typing import Iterator

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.sample_utils import make_sampler
from mlx_lm.generate import wired_limit, generation_stream
# v3 generalization: model-family-agnostic mask construction.
# Direct Qwen3.5 import kept as legacy fallback (see mask_adapter._qwen35_masks).
from app.backend.mask_adapter import get_masks as _get_masks_for_family
from app.core.model_family import ModelFamily, detect_family as _detect_family

from app.backend.adapter import GenerationChunk
from app.innovation.n1_ctrsp.state_persistence import CTRSPManager, CachedModelState
from app.innovation.n4_layer_skip.skip_controller import SkipController
from app.innovation.lookahead.lookahead_decode import (
    NGramPredictor, LookaheadStats,
    save_gdn_states, restore_gdn_states, rollback_fa_cache,
)
from app.engine.quality_monitor import QualityMonitor
from app.core.logging import get_logger

logger = get_logger(__name__)


def _build_escape_hint(prompt_tokens: list[int], tokenizer) -> str:
    """Return a task-aware transition string to inject after </think>.

    When the think-loop-escape fires, a bare "</think>\\n\\n" often lets the
    model continue in scratch-work style (bullets, "Wait, ...") instead of
    emitting a clean final answer. A task-typed hint like "Final JSON:" or
    "Final Code:" provides a strong transition that nudges the model into
    answer mode. The hint is derived from the tail of the prompt (last user
    message) which carries the actual task directive.
    """
    try:
        tail = prompt_tokens[-512:] if len(prompt_tokens) > 512 else prompt_tokens
        p = tokenizer.decode(tail).lower()
    except Exception:
        return "\n\n**Final Answer:**\n\n"

    if "json" in p and ("array" in p or "extract" in p or "output" in p):
        return "\n\n**Final JSON:**\n\n```json\n"
    if any(k in p for k in ("implement", "function", "```python", "def ")):
        return "\n\n**Final Code:**\n\n```python\n"
    if "sql" in p and ("optim" in p or "rank" in p or "propos" in p):
        return "\n\n**Final Answer — 3 ranked optimizations:**\n\n"
    if "exactly" in p and "sentence" in p:
        return "\n\n**Final 5 sentences:**\n\n"
    if "translate" in p or "japanese" in p or "日本語" in p:
        return "\n\n**Final Translation:**\n\n"
    if any(k in p for k in ("review", "bugs", "risks")) and "```python" in p:
        return "\n\n**Final Review — bugs ranked by severity:**\n\n"
    if any(k in p for k in ("design", "architecture", "failure modes")):
        return "\n\n**Final Answer:**\n\n"
    return "\n\n**Final Answer:**\n\n"


def custom_forward(
    inputs: mx.array,
    lm_model,
    inner_model,
    cache: list,
    skip_controller: SkipController | None = None,
    family: ModelFamily = ModelFamily.QWEN_3_5,
) -> mx.array:
    """Custom forward pass with per-layer Innovation hooks.

    v3 generalization: mask construction now goes through `mask_adapter` so
    non-Qwen3.5 families (Gemma, Llama, Qwen 2.5/3.6, Mistral) can plug in.
    Qwen3.5 behavior is 1:1 identical to the previous direct-import code.

    Replaces lm(inputs, cache=cache) with a version that can:
    - N4 ALS: Skip low-impact layers for easy tokens
    - (future) X2 DPC: Swap layer weights per-layer
    """
    # Embedding
    hidden_states = inner_model.embed_tokens(inputs)

    # Build masks (family-aware)
    masks = _get_masks_for_family(family, inner_model, hidden_states, cache)
    fa_mask = masks.fa_mask
    ssm_mask = masks.ssm_mask

    # Layer loop with skip hooks
    for i, (layer, c) in enumerate(zip(inner_model.layers, cache)):
        # N4 ALS: check if this layer should be skipped
        if skip_controller and skip_controller.should_skip_layer(i):
            continue  # Skip this layer entirely — hidden_states passes through unchanged

        # Pure transformer families have no is_linear flag / no GDN → always fa_mask.
        is_linear = getattr(layer, "is_linear", False)
        if is_linear and ssm_mask is not None:
            mask = ssm_mask
        else:
            mask = fa_mask
        hidden_states = layer(hidden_states, mask=mask, cache=c)

    # Final norm
    hidden_states = inner_model.norm(hidden_states)

    # LM head
    if lm_model.args.tie_word_embeddings:
        logits = inner_model.embed_tokens.as_linear(hidden_states)
    else:
        logits = lm_model.lm_head(hidden_states)

    return logits


def inject_ctrsp_state(
    cache: list,
    cached_state: CachedModelState,
    model_layers: list,
    target_dtype=None,
) -> int:
    """Inject cached GatedDeltaNet state AND FA KV cache into model cache.

    Returns number of tokens that can be skipped (the cached prefix length).
    """
    if target_dtype is None:
        target_dtype = mx.bfloat16

    n_prefix = cached_state.tokens_processed

    # Inject GDN recurrent states + set position tracking
    gdn_idx = 0
    for i, layer in enumerate(model_layers):
        if layer.is_linear and gdn_idx < len(cached_state.gdn_states):
            recurrent, conv = cached_state.gdn_states[gdn_idx]
            cache[i][0] = conv.astype(target_dtype)
            cache[i][1] = recurrent.astype(target_dtype)
            # ArraysCache tracks position via lengths — not needed for
            # standard inference (lengths is None unless batched with padding)
            gdn_idx += 1

    # Inject FA KV caches with correct offset
    fa_idx = 0
    fa_kv_list = cached_state.kv_cache_keys  # Stores (keys, values, offset) tuples
    for i, layer in enumerate(model_layers):
        if not layer.is_linear and fa_idx < len(fa_kv_list):
            keys, values, offset = fa_kv_list[fa_idx]
            kv_cache = cache[i]
            kv_cache.keys = keys.astype(target_dtype)
            kv_cache.values = values.astype(target_dtype)
            kv_cache.offset = offset
            fa_idx += 1

    logger.info(
        "N1: Injected %d GDN states + %d FA KV caches (%d prefix tokens)",
        gdn_idx, fa_idx, n_prefix,
    )
    return cached_state.tokens_processed


def extract_ctrsp_state(
    cache: list,
    model_layers: list,
) -> tuple[list[tuple[mx.array, mx.array]], list[tuple[mx.array, mx.array, int]]]:
    """Extract GatedDeltaNet state AND FA KV cache from model cache.

    Returns: (gdn_states, fa_kv_caches)
    """
    gdn_states = []
    fa_kv_caches = []

    for i, layer in enumerate(model_layers):
        if layer.is_linear:
            conv_state = cache[i][0]
            recurrent_state = cache[i][1]
            if conv_state is not None and recurrent_state is not None:
                gdn_states.append((
                    recurrent_state.astype(mx.float32),
                    conv_state.astype(mx.float32),
                ))
        else:
            kv_cache = cache[i]
            if kv_cache.keys is not None:
                fa_kv_caches.append((
                    kv_cache.keys.astype(mx.float32),
                    kv_cache.values.astype(mx.float32),
                    kv_cache.offset,
                ))

    return gdn_states, fa_kv_caches


def generate_with_innovations(
    model,
    tokenizer,
    prompt_tokens: list[int],
    max_tokens: int = 256,
    temperature: float = 0.0,
    ctrsp_manager: CTRSPManager | None = None,
    ctrsp_prompt_hash: str | None = None,
    ctrsp_prefix_len: int = 0,
    ctrsp_model_name: str = "",
    skip_controller: SkipController | None = None,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    lookahead_k: int = 0,
    quality_monitor: QualityMonitor | None = None,
    progressive_chunk_size: int = 1024,
    ssd_predictor=None,           # Structural Skip Decoding
    model_family: ModelFamily | None = None,  # auto-detect if None
) -> Iterator[GenerationChunk]:
    """Generate tokens with Innovation hooks integrated.

    This is the core generation loop that replaces mlx_lm.stream_generate
    when Innovation techniques need model-level access.

    Args:
        kv_bits: Quantize KV cache to this many bits (4 or 8). Reduces memory,
                 may improve cache locality. None = full precision.
    """
    lm = model.language_model if hasattr(model, 'language_model') else model
    inner_model = lm.model if hasattr(lm, 'model') else lm
    layers = inner_model.layers

    # v3 generalization: resolve model family (default Qwen 3.5 for regression safety)
    if model_family is None:
        try:
            from app.core.config import get_settings as _gs_mf
            _mcfg = getattr(_gs_mf(), "model", None)
            _fam_str = getattr(_mcfg, "family", None) if _mcfg is not None else None
            if _fam_str and _fam_str != "auto":
                model_family = ModelFamily.from_string(_fam_str)
            else:
                # Auto: use main model path; if unavailable, fall back to Qwen3.5
                _path = getattr(_mcfg, "main_path", "") if _mcfg is not None else ""
                model_family = _detect_family(_path) if _path else ModelFamily.QWEN_3_5
        except Exception:
            model_family = ModelFamily.QWEN_3_5

    # Create cache (optionally quantized — disabled when CTRSP restores state)
    # Family-aware: pure transformers use only QuantizedKVCache (no GDN layers).
    use_quantized_kv = kv_bits is not None and not (ctrsp_manager and ctrsp_prompt_hash)
    if use_quantized_kv:
        from mlx_lm.models.cache import QuantizedKVCache, ArraysCache as _AC
        cache = []
        for layer in layers:
            # Pure transformers have no is_linear flag → all layers get KV cache
            if getattr(layer, "is_linear", False):
                cache.append(_AC(size=2))
            else:
                cache.append(QuantizedKVCache(group_size=kv_group_size, bits=kv_bits))
        logger.debug("Using quantized KV cache (bits=%d, family=%s)", kv_bits, model_family.value)
    else:
        cache = lm.make_cache() if hasattr(lm, 'make_cache') else model.make_cache()

    # ── N1: CTRSP State Injection ─────────────────────────
    tokens_to_skip = 0
    if ctrsp_manager and ctrsp_prompt_hash:
        cached = ctrsp_manager.get_cached_state(ctrsp_prompt_hash, ctrsp_model_name)
        if cached:
            tokens_to_skip = inject_ctrsp_state(cache, cached, layers)
            logger.info("N1: Skipping %d tokens of prefill via cached state", tokens_to_skip)

    # ── Wire model memory + use dedicated GPU stream ────
    # wired_limit pins model weights to prevent swapping (mlx_lm best practice).
    # Wrap ALL downstream work in try/finally so the wired context is always
    # released — otherwise an early generator abandonment or mid-loop exception
    # leaks pinned memory until interpreter exit.
    _wired_ctx = wired_limit(model, [generation_stream])
    _wired_ctx.__enter__()
    try:
        yield from _run_generation(
            lm=lm, inner_model=inner_model, layers=layers, cache=cache,
            prompt_tokens=prompt_tokens, tokens_to_skip=tokens_to_skip,
            ctrsp_manager=ctrsp_manager, ctrsp_prompt_hash=ctrsp_prompt_hash,
            ctrsp_prefix_len=ctrsp_prefix_len, ctrsp_model_name=ctrsp_model_name,
            tokenizer=tokenizer, max_tokens=max_tokens, temperature=temperature,
            skip_controller=skip_controller, lookahead_k=lookahead_k,
            quality_monitor=quality_monitor, ssd_predictor=ssd_predictor,
            progressive_chunk_size=progressive_chunk_size,
            model_family=model_family,
        )
    finally:
        _wired_ctx.__exit__(None, None, None)


def _run_generation(
    *,
    lm, inner_model, layers, cache,
    prompt_tokens: list[int],
    tokens_to_skip: int,
    ctrsp_manager: CTRSPManager | None,
    ctrsp_prompt_hash: str | None,
    ctrsp_prefix_len: int,
    ctrsp_model_name: str,
    tokenizer,
    max_tokens: int,
    temperature: float,
    skip_controller: SkipController | None,
    lookahead_k: int,
    quality_monitor: QualityMonitor | None,
    progressive_chunk_size: int,
    ssd_predictor=None,
    model_family: ModelFamily = ModelFamily.QWEN_3_5,
) -> Iterator[GenerationChunk]:
    """Inner generation body. Runs inside wired_limit context."""
    # ── Prefill ───────────────────────────────────────────
    prefill_start = time.perf_counter()
    prompt = mx.array(prompt_tokens)

    def _chunked_prefill(tokens_1d: mx.array) -> mx.array:
        """Run prefill in chunks to stay within Metal buffer limits.
        Returns logits of the last chunk.
        """
        total = tokens_1d.shape[0]
        if total == 0:
            raise ValueError("Empty tokens in _chunked_prefill")
        if total <= progressive_chunk_size:
            out = lm(tokens_1d[None], cache=cache)
            mx.eval(out)
            return out
        logger.info("Progressive prefill: %d tokens, chunk=%d", total, progressive_chunk_size)
        start = 0
        out = None
        while start < total:
            end = min(start + progressive_chunk_size, total)
            out = lm(tokens_1d[start:end][None], cache=cache)
            mx.eval(out)
            # Clear transient Metal allocations between chunks to avoid OOM
            # (preserves cache since wired_limit pins model weights)
            mx.clear_cache()
            start = end
            if start < total:
                logger.info("  prefill %d/%d (%.0f%%)", start, total, start / total * 100)
        return out

    if tokens_to_skip > 0 and tokens_to_skip < len(prompt_tokens):
        # CTRSP RESTORE path: process only new tokens after cached prefix
        new_tokens = prompt[tokens_to_skip:]
        logits = _chunked_prefill(new_tokens)

    elif tokens_to_skip > 0 and tokens_to_skip == len(prompt_tokens):
        # CTRSP EXACT MATCH: the full prompt is exactly the cached prefix.
        # Cache state is at position N; we need logits for the last token to
        # seed the sampler. Re-run just the final token with offset rollback
        # of 1 so the FA KV cache doesn't double-count.
        logger.info("N1: exact prefix match — recomputing last token for logits")
        for i, layer in enumerate(layers):
            if not layer.is_linear:
                kv = cache[i]
                if kv.offset > 0:
                    kv.offset = max(0, kv.offset - 1)
        logits = _chunked_prefill(prompt[-1:])

    elif ctrsp_manager and ctrsp_prompt_hash and tokens_to_skip == 0 and ctrsp_prefix_len > 0:
        # CTRSP SAVE path: prefill system prompt separately, save state, then continue
        sys_part = prompt[:ctrsp_prefix_len]
        _logits_sys = _chunked_prefill(sys_part)

        # Save state right after system prompt (before question tokens get mixed in)
        gdn_states, fa_kv_caches = extract_ctrsp_state(cache, layers)
        if gdn_states:
            ctrsp_manager.save_state(
                prompt_hash=ctrsp_prompt_hash,
                model_name=ctrsp_model_name,
                gdn_states=gdn_states,
                kv_cache_keys=fa_kv_caches,
                tokens_processed=ctrsp_prefix_len,
                position_offset=ctrsp_prefix_len,
            )
            logger.info("N1: Saved state after %d system prompt tokens", ctrsp_prefix_len)

        # Continue prefill with remaining tokens (question)
        remaining = prompt[ctrsp_prefix_len:]
        if remaining.size > 0:
            logits = _chunked_prefill(remaining)
        else:
            logits = _logits_sys

    else:
        # Standard full prefill — chunked for long prompts
        logits = _chunked_prefill(prompt)

    prefill_time = time.perf_counter() - prefill_start

    # ── Sampling setup ────────────────────────────────────
    sampler = make_sampler(temp=temperature)

    # ── X5-R: Compile model forward for Metal kernel fusion ─
    # Note: mx.compile can't accept cache objects directly, so we
    # compile the model itself rather than the full decode step.
    # This still fuses the layer computations within each forward pass.
    try:
        compiled_model = mx.compile(lm)
        use_compiled = True
        logger.debug("X5-R: Model compiled with mx.compile")
    except Exception:
        compiled_model = lm
        use_compiled = False

    # ── Decode loop ───────────────────────────────────────
    token = sampler(logits[:, -1, :])
    tic = time.perf_counter()

    # Build EOS set
    eos_ids = tokenizer.eos_token_id
    if not isinstance(eos_ids, (list, tuple, set)):
        eos_ids = [eos_ids]
    eos_set = set(eos_ids)

    # Lookahead setup
    use_lookahead = lookahead_k > 0
    ngram = NGramPredictor(n=3) if use_lookahead else None
    la_stats = LookaheadStats() if use_lookahead else None
    generated_ids: list[int] = list(prompt_tokens)

    # Streaming detokenizer — handles BPE/SPM boundaries correctly
    # (multi-byte UTF-8 that spans multiple tokens) and avoids re-decoding
    # from scratch each call. Fallback to per-token decode if unavailable.
    _detok = getattr(tokenizer, "detokenizer", None)
    if _detok is not None:
        try:
            _detok.reset()
        except Exception:
            _detok = None

    def _streaming_text(tok_id: int) -> str:
        if _detok is not None:
            try:
                _detok.add_token(tok_id)
                return _detok.last_segment
            except Exception:
                pass
        return tokenizer.decode([tok_id])

    # Cache peak memory — Metal API call per token is unnecessary; we only
    # need a fresh reading for the emitted GenerationChunk periodically.
    _last_peak_mem_gb = mx.get_peak_memory() / 1e9
    _peak_mem_check_every = 10

    # ── Thinking-aware token budget ──────────────────────────────
    # Qwen3.5 wraps chain-of-thought in <think>...</think>. Token IDs
    # are resolved at the Qwen tokenizer level (single tokens: 248068 / 248069).
    # Inside a think block, we do NOT count tokens toward max_tokens (the
    # user-facing answer budget). This prevents benign "thinking ran out of
    # budget before final answer" failures without increasing answer length.
    # A safety cap on thinking length defaults to 32768 to prevent runaway loops.
    from app.core.config import get_settings as _get_settings_for_budget
    _cfg = _get_settings_for_budget()
    _max_thinking = getattr(_cfg.generation, "max_thinking_tokens", 32768)
    # v3 generalization: use family profile to decide think-aware handling.
    # Pure transformer families (Llama/Gemma/Mistral) never enter think mode
    # and saving the tokenizer.encode roundtrip is a nice microwin too.
    try:
        from app.core.model_family import get_profile as _get_fam_profile
        _fam_profile = _get_fam_profile(model_family)
    except Exception:
        _fam_profile = None

    if _fam_profile is not None and not _fam_profile.supports_thinking:
        # Family declared as non-thinking — skip think logic entirely
        _think_open_id = None
        _think_close_id = None
    else:
        # Thinking-capable family (Qwen3.5/3.6): resolve tokens from tokenizer
        _open_tok = _fam_profile.think_open_token if _fam_profile else "<think>"
        _close_tok = _fam_profile.think_close_token if _fam_profile else "</think>"
        try:
            _think_open_id = tokenizer.encode(_open_tok, add_special_tokens=False)
            _think_close_id = tokenizer.encode(_close_tok, add_special_tokens=False)
            _think_open_id = _think_open_id[0] if len(_think_open_id) == 1 else None
            _think_close_id = _think_close_id[0] if len(_think_close_id) == 1 else None
        except Exception:
            _think_open_id = None
            _think_close_id = None
    # Initial state: Qwen3.5's chat template prefills `<think>\n` at the end
    # of the prompt when enable_thinking=True, so the model starts inside a
    # think block even though it never emits the <think> token itself. Detect
    # by scanning the tail of prompt_tokens for <think> without a matching </think>.
    in_think_block = False
    if _think_open_id is not None and _think_close_id is not None:
        tail = prompt_tokens[-64:] if len(prompt_tokens) > 64 else prompt_tokens
        last_open = None
        last_close = None
        for _i, _t in enumerate(tail):
            if _t == _think_open_id:
                last_open = _i
            elif _t == _think_close_id:
                last_close = _i
        if last_open is not None and (last_close is None or last_close < last_open):
            in_think_block = True
            logger.debug("Generation starts inside <think> block (prompt prefilled by chat template)")
    n_thinking = 0

    # Aggressive thinking-phase loop detector. Fires during in_think_block only.
    # Kept at ngram=6/min_window=48 — tighter values (5/24) false-positive on
    # healthy bullet lists (e.g., the "*   **Researchers:** ... *   **Fields:**"
    # pattern at the start of the logic-puzzle task).
    from app.engine.quality_monitor import QualityMonitor as _QM
    _think_monitor = _QM(
        ngram_size=6,
        history_size=128,
        max_ngram_repeats=3,
        min_window_for_detection=48,
        check_every_n_tokens=8,
    )

    # (Phrase-canary removed: first-token matching of "Wait"/"Actually" caused
    # false positives on healthy logical reconsideration in puzzle-like tasks.
    # The n-gram detector + task-aware escape hint are sufficient.)

    n = 0
    while n < max_tokens:
        token_id = token.item()

        # Track think-block state BEFORE counting/yielding
        if _think_open_id is not None and token_id == _think_open_id:
            in_think_block = True
        # We mark closing AFTER the token is counted to the thinking side
        # (so the closing tag itself is part of the thinking stream).

        # Thinking cap — force </think> injection when we've spent
        # enough thinking budget, rather than terminating.
        if in_think_block and n_thinking >= _max_thinking and _think_close_id is not None:
            logger.warning(
                "Thinking cap (%d) reached; injecting </think> to force answer",
                _max_thinking,
            )
            # Inject </think> + task-aware hint (same mechanism as loop-escape)
            _cap_tokens = [_think_close_id]
            try:
                _hint = _build_escape_hint(prompt_tokens, tokenizer)
                _cap_tokens.extend(tokenizer.encode(_hint, add_special_tokens=False))
            except Exception:
                pass
            _close_text = tokenizer.decode(_cap_tokens)
            for _cid in _cap_tokens:
                generated_ids.append(_cid)
                n_thinking += 1
            in_think_block = False
            yield GenerationChunk(
                text=_close_text, token=_cap_tokens[-1],
                finish_reason=None,
                prompt_tokens=len(prompt_tokens),
                generation_tokens=n + 1,
                prompt_tps=len(prompt_tokens) / prefill_time if prefill_time > 0 else 0,
                generation_tps=(n + 1) / max(time.perf_counter() - tic, 1e-6),
                peak_memory_gb=_last_peak_mem_gb,
            )
            if quality_monitor is not None:
                try: quality_monitor.reset()
                except Exception: pass
            inject = mx.array([_cap_tokens])
            new_logits = lm(inject, cache=cache)
            mx.eval(new_logits)
            logits = new_logits
            token = sampler(logits[:, -1, :])
            continue

        if token_id in eos_set:
            break

        # === Aggressive thinking-phase loop detection ===
        # Fires much earlier than the main QualityMonitor (ngram=6, repeats=3)
        # so we don't burn thousands of tokens on thinking loops before the
        # main detector kicks in. Only active while in a think block.
        _think_anomaly = None
        if in_think_block:
            _think_anomaly = _think_monitor.record(token_id)

        # === Quality monitoring: early termination on repetition loops ===
        if quality_monitor is not None:
            anomaly = quality_monitor.record(token_id)
            # Promote think-phase anomaly to the normal escape path
            if _think_anomaly is not None and _think_anomaly.detected and not anomaly.detected:
                anomaly = _think_anomaly
            if anomaly.detected:
                # THINK-LOOP ESCAPE: if we're still in a think block and the
                # monitor detects a loop, inject </think>\n\n to force the
                # model into the answer phase with a strong state transition.
                # Only attempted once — a second anomaly after escape is a
                # real quality failure and stops generation.
                if in_think_block and _think_close_id is not None:
                    logger.warning(
                        "Think-loop detected (%s); injecting </think>\\n\\n to rescue answer",
                        anomaly.reason,
                    )
                    # Inject [</think>, <task-aware hint>] as a multi-token
                    # prefix to transition into answer mode. Plain "\n\n" lets
                    # the model continue in scratchwork style; a typed hint
                    # ("Final JSON:", "Final Code:", etc.) nudges it into the
                    # required output shape.
                    _post_think_tokens = [_think_close_id]
                    try:
                        _hint = _build_escape_hint(prompt_tokens, tokenizer)
                        _hint_ids = tokenizer.encode(_hint, add_special_tokens=False)
                        _post_think_tokens.extend(_hint_ids)
                    except Exception:
                        try:
                            _post_think_tokens.extend(
                                tokenizer.encode("\n\n", add_special_tokens=False)
                            )
                        except Exception:
                            pass
                    _post_text = tokenizer.decode(_post_think_tokens)
                    for _pid in _post_think_tokens:
                        generated_ids.append(_pid)
                        n_thinking += 1
                    in_think_block = False
                    yield GenerationChunk(
                        text=_post_text, token=_post_think_tokens[-1],
                        finish_reason=None,
                        prompt_tokens=len(prompt_tokens),
                        generation_tokens=n + 1,
                        prompt_tps=len(prompt_tokens) / prefill_time if prefill_time > 0 else 0,
                        generation_tps=(n + 1) / max(time.perf_counter() - tic, 1e-6),
                        peak_memory_gb=_last_peak_mem_gb,
                    )
                    try:
                        quality_monitor.reset()
                    except Exception:
                        pass
                    # Feed injected tokens through the model for cache consistency
                    inject = mx.array([_post_think_tokens])
                    new_logits = lm(inject, cache=cache)
                    mx.eval(new_logits)
                    logits = new_logits
                    token = sampler(logits[:, -1, :])
                    continue

                logger.warning(
                    "Quality anomaly detected, stopping generation: %s (%s)",
                    anomaly.reason, anomaly.detail,
                )
                # Emit final chunk and stop
                yield GenerationChunk(
                    text="",
                    token=token_id,
                    finish_reason="stop",
                    prompt_tokens=len(prompt_tokens),
                    generation_tokens=n + 1,
                    prompt_tps=len(prompt_tokens) / prefill_time if prefill_time > 0 else 0,
                    generation_tps=(n + 1) / max(time.perf_counter() - tic, 1e-6),
                    peak_memory_gb=_last_peak_mem_gb,
                )
                return

        # Yield this token
        elapsed = time.perf_counter() - tic
        tps = (n + 1) / elapsed if elapsed > 0 else 0
        text = _streaming_text(token_id)

        yield GenerationChunk(
            text=text,
            token=token_id,
            finish_reason=None,
            prompt_tokens=len(prompt_tokens),
            generation_tokens=n + 1,
            prompt_tps=len(prompt_tokens) / prefill_time if prefill_time > 0 else 0,
            generation_tps=tps,
            peak_memory_gb=_last_peak_mem_gb,
        )

        generated_ids.append(token_id)
        # Thinking-aware counting: tokens inside <think>...</think> use the
        # separate _max_thinking budget and do NOT consume the user's answer
        # budget (max_tokens).
        if in_think_block:
            n_thinking += 1
        else:
            n += 1

        # Close think block AFTER counting the closing tag.
        if _think_close_id is not None and token_id == _think_close_id and in_think_block:
            in_think_block = False

        # Refresh peak memory reading periodically (not per token)
        if (n + n_thinking) % _peak_mem_check_every == 0:
            _last_peak_mem_gb = mx.get_peak_memory() / 1e9

        # ── Lookahead: try to batch-verify multiple tokens ──
        if use_lookahead and n >= 10:  # Need some context for n-gram
            ngram.update(generated_ids[-50:])  # Update n-gram table

            # SSD: try structural prediction first (highest confidence)
            # SSD predictor returns single high-confidence next token, OR None.
            # If SSD has a prediction, prepend it to the draft chain.
            ssd_pred = None
            if ssd_predictor is not None:
                try:
                    ssd_pred = ssd_predictor.predict(generated_ids)
                except Exception:
                    ssd_pred = None

            # n-gram chain (existing path)
            draft = ngram.predict(generated_ids[-3:], lookahead_k)

            # If SSD has a prediction, prepend it (then n-gram extends from it)
            if ssd_pred is not None:
                if not draft or draft[0] != ssd_pred:
                    draft = [ssd_pred] + (draft[:lookahead_k - 1] if draft else [])

            if len(draft) >= 1:
                la_stats.draft_attempts += 1
                la_stats.draft_tokens += len(draft)

                # Save state for rollback
                saved_gdn = save_gdn_states(cache, layers)
                saved_fa_offsets = {
                    i: cache[i].offset for i, l in enumerate(layers) if not l.is_linear
                }

                # Batch: [current_token, draft_1, draft_2, ...]
                batch = mx.array([[token_id] + draft])
                batch_logits = lm(batch, cache=cache)
                mx.eval(batch_logits)

                # Verify: check if model predictions match draft
                accepted = 0
                for j in range(len(draft)):
                    predicted = mx.argmax(batch_logits[0, j, :]).item()
                    if predicted == draft[j]:
                        accepted += 1
                    else:
                        break

                if accepted > 0:
                    # Accept tokens — yield them
                    la_stats.tokens_accepted += accepted
                    quality_stop = False
                    for j in range(accepted):
                        tok_id = draft[j]
                        if tok_id in eos_set:
                            break
                        # QualityMonitor: repetition loops can bypass main
                        # decode entirely when an n-gram predictor drafts
                        # exactly the loop — must check here too.
                        if quality_monitor is not None:
                            anomaly = quality_monitor.record(tok_id)
                            if anomaly.detected:
                                logger.warning(
                                    "Quality anomaly in lookahead: %s (%s)",
                                    anomaly.reason, anomaly.detail,
                                )
                                quality_stop = True
                                break
                        # Thinking-aware counting in lookahead path
                        if _think_open_id is not None and tok_id == _think_open_id:
                            in_think_block = True
                        if in_think_block:
                            n_thinking += 1
                        else:
                            n += 1
                        if _think_close_id is not None and tok_id == _think_close_id and in_think_block:
                            in_think_block = False
                        generated_ids.append(tok_id)
                        elapsed = time.perf_counter() - tic
                        tps = n / elapsed if elapsed > 0 else 0
                        yield GenerationChunk(
                            text=_streaming_text(tok_id),
                            token=tok_id,
                            finish_reason=None,
                            prompt_tokens=len(prompt_tokens),
                            generation_tokens=n,
                            prompt_tps=0,
                            generation_tps=tps,
                            peak_memory_gb=_last_peak_mem_gb,
                        )
                    if quality_stop:
                        yield GenerationChunk(
                            text="", token=tok_id, finish_reason="stop",
                            prompt_tokens=len(prompt_tokens),
                            generation_tokens=n,
                            prompt_tps=len(prompt_tokens) / prefill_time if prefill_time > 0 else 0,
                            generation_tps=n / max(time.perf_counter() - tic, 1e-6),
                            peak_memory_gb=_last_peak_mem_gb,
                        )
                        return

                # Rollback unaccepted tokens from cache
                rollback_count = len(draft) - accepted
                if rollback_count > 0:
                    restore_gdn_states(cache, saved_gdn)
                    rollback_fa_cache(cache, layers, rollback_count)
                    la_stats.full_rollbacks += 1

                    # Re-run with only accepted tokens to get correct state
                    if accepted > 0:
                        rerun = mx.array([[token_id] + draft[:accepted]])
                        logits = lm(rerun, cache=cache)
                        mx.eval(logits)
                    else:
                        logits = lm(mx.array([[token_id]]), cache=cache)
                        mx.eval(logits)

                    token = sampler(logits[:, -1, :])
                else:
                    # All accepted — next token is model's prediction after last draft
                    token = sampler(batch_logits[:, -1, :])

                la_stats.total_steps += 1
                la_stats.tokens_generated += 1 + accepted
                continue

        # Normal single-token decode (no lookahead or n-gram miss)
        logits = custom_forward(
            token.reshape(1, 1), lm, inner_model, cache,
            skip_controller=skip_controller,
            family=model_family,
        )
        # Feed top-1 softmax confidence to N4 skip_controller so the NEXT
        # token's layer-skip decision reflects the current token's difficulty.
        if skip_controller is not None:
            try:
                probs = mx.softmax(logits[:, -1, :], axis=-1)
                top1 = mx.max(probs, axis=-1)
                mx.eval(top1)
                skip_controller.set_last_confidence(top1.item())
            except Exception:
                pass
        token = sampler(logits[:, -1, :])
        if la_stats:
            la_stats.total_steps += 1
            la_stats.tokens_generated += 1

    # Final chunk
    elapsed = time.perf_counter() - tic
    finish = "stop" if token.item() in eos_set else "length"

    if la_stats and la_stats.total_steps > 0:
        logger.info(
            "Lookahead: %d steps, %d tokens (%.1f tok/step), %d accepted (%.0f%% rate), %d rollbacks",
            la_stats.total_steps, la_stats.tokens_generated,
            la_stats.tokens_per_step,
            la_stats.tokens_accepted,
            la_stats.acceptance_rate * 100,
            la_stats.full_rollbacks,
        )

    # Flush any residual segment the streaming detokenizer was holding
    trailing = ""
    if _detok is not None:
        try:
            _detok.finalize()
            trailing = _detok.last_segment
        except Exception:
            trailing = ""

    yield GenerationChunk(
        text=trailing,
        token=token.item(),
        finish_reason=finish,
        prompt_tokens=len(prompt_tokens),
        generation_tokens=n,
        prompt_tps=len(prompt_tokens) / prefill_time if prefill_time > 0 else 0,
        generation_tps=n / elapsed if elapsed > 0 else 0,
        peak_memory_gb=mx.get_peak_memory() / 1e9,
    )
