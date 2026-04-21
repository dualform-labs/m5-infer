"""Innovation Executor — bridges GeneratePlan decisions to actual inference.

This is the critical integration layer that connects:
  GeneratePlan (what to do) → custom_generate (Innovation-aware inference)

All Innovation techniques are applied through custom_generate.generate_with_innovations(),
NOT through the basic backend.generate_stream() wrapper.
"""

from __future__ import annotations

import time
from typing import Iterator

from app.backend.adapter import GenerationChunk
from app.backend.mlx_text_backend import MLXTextBackend
from app.backend.custom_generate import generate_with_innovations
from app.engine.cache_manager import CacheManager
from app.engine.long_gen_optimizer import LongGenerationOptimizer
from app.innovation.feature_flags import (
    is_n1_ctrsp_enabled,
    is_x2_dpc_enabled,
    is_x4_context_fold_enabled,
    is_n4_als_enabled,
)
from app.core.config import get_settings
from app.innovation.n1_ctrsp.session_integration import (
    get_ctrsp_manager,
    extract_system_prompt_tokens,
)
from app.innovation.x4_context_fold.schema_analyzer import analyze_tool_schemas
from app.innovation.x2_precision.confidence_router import ConfidenceRouter
from app.innovation.n4_layer_skip.profiler import LayerProfiler
from app.innovation.n4_layer_skip.skip_controller import SkipController
from app.engine.quality_monitor import QualityMonitor
from app.engine.context_redundancy import analyze_prompt_redundancy
from app.innovation.tpc import get_tpc_cache, get_background_compiler
from app.planner.plan_types import GeneratePlan
from app.core.logging import get_logger, MetricsLogger

logger = get_logger(__name__)
metrics_logger = MetricsLogger()


class InnovationExecutor:
    """Executes inference via custom_generate with all Innovation hooks active."""

    def __init__(
        self,
        backend: MLXTextBackend,
        cache_manager: CacheManager,
    ):
        self._backend = backend
        self._cache_manager = cache_manager
        self._long_gen_opt = LongGenerationOptimizer()
        self._dpc_router: ConfidenceRouter | None = None
        self._x2_shadow = None  # ShadowWeights instance when aggressive/extreme mode
        self._skip_controller: SkipController | None = None
        self._draft_model = None  # Optional draft model for speculative decoding
        self._draft_tokenizer = None
        self._stats = {
            "requests": 0,
            "ctrsp_hits": 0,
            "ctrsp_saves": 0,
            "context_fold_applied": 0,
            "n4_layers_skipped": 0,
        }

    def set_draft_model(self, draft_model, draft_tokenizer) -> None:
        """Attach a draft model for speculative decoding.

        Draft model must share the tokenizer with the main model.
        Auto-detects architecture: fast path for pure Transformers,
        slow path (with warning) for hybrid models.
        """
        main_tokenizer = self._backend.get_tokenizer()
        # Verify tokenizer compatibility (vocab_size check)
        if hasattr(main_tokenizer, "vocab_size") and hasattr(draft_tokenizer, "vocab_size"):
            if main_tokenizer.vocab_size != draft_tokenizer.vocab_size:
                logger.warning(
                    "Draft tokenizer vocab mismatch (main=%d, draft=%d) — NOT setting draft",
                    main_tokenizer.vocab_size, draft_tokenizer.vocab_size,
                )
                return

        self._draft_model = draft_model
        self._draft_tokenizer = draft_tokenizer

        # Detect architecture
        from app.innovation.speculative.draft_speculative import _is_pure_transformer
        main_model = self._backend._model
        main_lm = main_model.language_model if hasattr(main_model, 'language_model') else main_model
        main_inner = main_lm.model if hasattr(main_lm, 'model') else main_lm
        is_pure = _is_pure_transformer(main_inner)

        if is_pure:
            logger.info("Speculative decoding enabled (FAST path — pure Transformer)")
        else:
            logger.warning(
                "Speculative decoding enabled (SLOW path — hybrid architecture). "
                "May NOT improve speed; consider disabling for Qwen3.5 etc."
            )

    def initialize_innovations(self) -> None:
        # ── X2 DPC full activation (mode-gated) ──
        feat = get_settings().runtime.active_features()
        if is_x2_dpc_enabled():
            self._dpc_router = ConfidenceRouter()
            logger.info("Innovation Executor: DPC confidence router initialized")

            # Generate 2-bit shadow weights when mode demands it
            # (aggressive/extreme). Memory cost ~2.8 GB.
            if feat.x2_shadow_enabled and self._backend._model is not None:
                try:
                    from app.innovation.x2_precision.shadow_weights import ShadowWeights
                    shadow = ShadowWeights()
                    shadow.generate(self._backend._model)
                    if shadow.is_ready():
                        self._x2_shadow = shadow
                        logger.info(
                            "X2 DPC: 2-bit shadow weights ready (%.2f GB)",
                            shadow.memory_bytes() / 1e9,
                        )
                    else:
                        logger.warning("X2 DPC: shadow generation produced no layers")
                except Exception:
                    logger.exception("X2 DPC: shadow generation failed")

        if is_n4_als_enabled() and self._backend._model is not None:
            try:
                profiler = LayerProfiler()
                model = self._backend._model
                lm = model.language_model if hasattr(model, 'language_model') else model
                inner = lm.model if hasattr(lm, 'model') else lm

                # Profile with a realistic-length sample (64 tokens) sampled
                # from the natural language id range so layer similarity
                # reflects typical inference workload, not a 3-token degenerate
                # case where norms/residuals are abnormally small.
                import mlx.core as mx
                tokenizer = self._backend.get_tokenizer()
                try:
                    seed_text = (
                        "The following is a multi-turn conversation between a user "
                        "and an AI assistant about various technical topics including "
                        "machine learning, software engineering, and mathematics."
                    )
                    seed_ids = tokenizer.encode(seed_text)[:64]
                    if len(seed_ids) < 16:
                        seed_ids = list(range(16, 16 + 64))
                except Exception:
                    seed_ids = list(range(16, 16 + 64))
                sample = inner.embed_tokens(mx.array([seed_ids]))
                profiler.profile(lm, sample)

                if profiler.skip_count() > 0:
                    self._skip_controller = SkipController(profiler)
                    self._skip_controller.activate()
                    logger.info(
                        "N4 ALS: %d/%d layers skippable (%.0f%%, profile len=%d)",
                        profiler.skip_count(),
                        len(inner.layers),
                        profiler.skip_count() / len(inner.layers) * 100,
                        len(seed_ids),
                    )
                else:
                    logger.info("N4 ALS: no layers skippable at threshold")
            except Exception:
                logger.exception("N4 ALS: profiling failed")

    def set_backend(self, backend) -> None:
        self._backend = backend
        self.initialize_innovations()

    def execute(
        self,
        plan: GeneratePlan,
        messages: list[dict],
        prompt_tokens: list[int],
        tools: list[dict] | None = None,
        request_id: str = "",
        session_id: str = "",
    ) -> Iterator[GenerationChunk]:
        """Execute inference with ALL Innovation techniques applied.

        Uses custom_generate.generate_with_innovations() which has
        model-level hooks for N1 CTRSP (state injection/extraction).
        """
        self._stats["requests"] += 1
        start_time = time.perf_counter()

        # ── Pre-generation: Context Folding analysis ──────────
        # NOTE: analyze_tool_schemas is currently advisory-only (result is not
        # applied back to the prompt). Kept behind a debug log to avoid the
        # per-request tokenizer cost when nothing consumes the result.
        if logger.isEnabledFor(10) and plan.enable_context_fold and is_x4_context_fold_enabled() and tools:
            fold_analysis = analyze_tool_schemas(tools, self._backend.get_tokenizer())
            if fold_analysis.worth_folding:
                self._stats["context_fold_applied"] += 1
                logger.debug("X4 advisory: tool schemas are foldable")

        # ── Long-gen optimizer ────────────────────────────────
        if plan.speed.prefer_long_generation_optimizer:
            self._long_gen_opt.activate(plan.effective_max_output_tokens)

        # ── Prepare CTRSP parameters ─────────────────────────
        ctrsp_manager = None
        ctrsp_hash = None
        ctrsp_prefix_len = 0

        model_id = self._backend.get_model_name() if hasattr(self._backend, "get_model_name") else ""

        if plan.enable_ctrsp and is_n1_ctrsp_enabled():
            tokenizer = self._backend.get_tokenizer()

            # TPC fast path: raw-bytes hash of system content, short-circuits
            # the expensive extract_system_prompt_tokens tokenization when we
            # have seen this system prompt before.
            sys_raw = None
            try:
                sys_raw_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
                if sys_raw_parts:
                    sys_raw = "\n".join(sys_raw_parts)
            except Exception:
                sys_raw = None

            tpc = get_tpc_cache()
            tpc_entry = tpc.lookup(sys_raw) if sys_raw else None

            sys_tokens = None
            if tpc_entry is not None:
                sys_tokens = list(tpc_entry.token_ids)
            else:
                sys_tokens = extract_system_prompt_tokens(messages, tokenizer)

            # T7 CPSB — CTRSP Short-Prompt Bypass.
            # Saving state via np.savez_compressed on a ~52 MB per-turn payload
            # blocks the request thread for 500-2000 ms (CPU-bound gzip) and
            # dominates TTFT when the system prompt is short. For prompts below
            # this threshold, the savings on re-prefill do not outweigh the
            # per-request compression cost, so we bypass CTRSP entirely.
            _CTRSP_MIN_SYS_TOKENS = 1024
            if sys_tokens is not None:
                if (
                    len(sys_tokens) >= _CTRSP_MIN_SYS_TOKENS
                    and len(sys_tokens) <= len(prompt_tokens)
                    and prompt_tokens[: len(sys_tokens)] == list(sys_tokens)
                ):
                    ctrsp_manager = get_ctrsp_manager()
                    if tpc_entry is not None and tpc_entry.ctrsp_token_hash:
                        ctrsp_hash = tpc_entry.ctrsp_token_hash
                    else:
                        ctrsp_hash = ctrsp_manager.compute_prompt_hash(sys_tokens)
                    ctrsp_prefix_len = len(sys_tokens)

                    cached = ctrsp_manager.get_cached_state(ctrsp_hash, model_id)

                    # Register/refresh TPC entry so WARM calls skip tokenization
                    if sys_raw:
                        tpc.store(sys_raw, sys_tokens, ctrsp_token_hash=ctrsp_hash)

                    # Redundancy analysis moves OFF critical path to background.
                    # (Previously blocking prefill for 2-3 sec on tool_heavy COLD.)
                    if (
                        cached is None
                        and len(sys_tokens) > 1024
                        and (tpc_entry is None or not tpc_entry.redundancy_scanned)
                    ):
                        _sys_tokens_snapshot = list(sys_tokens)
                        _sys_raw_snapshot = sys_raw
                        def _bg_redundancy():
                            report = analyze_prompt_redundancy(_sys_tokens_snapshot)
                            if report.recommendations:
                                for rec in report.recommendations:
                                    # T2 CPP — advisory is already background, but INFO
                                    # still hits logging lock. DEBUG suffices for observability.
                                    logger.debug("Context advisory (bg): %s", rec)
                            if _sys_raw_snapshot:
                                get_tpc_cache().mark_redundancy_scanned(_sys_raw_snapshot)
                        get_background_compiler().submit("redundancy_scan", _bg_redundancy)
                else:
                    logger.debug(
                        "CTRSP skipped: templated system tokens (%d) not a prefix of prompt (%d)",
                        len(sys_tokens), len(prompt_tokens),
                    )

        # ── Get the raw model for custom_generate ─────────────
        model = self._backend._model
        tokenizer = self._backend.get_tokenizer()

        if model is None:
            raise RuntimeError("Model not loaded")

        # ── MTAB observation hook (Phase D closing, TPC-gated) ──
        # Record prefix match statistics without changing decode behavior.
        # TPC: if we've already observed this exact system content, skip
        # the sha256 over 12K tokens on the critical path. For new content,
        # dispatch the observation to the background worker.
        try:
            from app.core.config import get_settings as _gs2
            if _gs2().runtime.active_features().mtab_enabled:
                _tpc_already = (
                    'tpc_entry' in locals()
                    and tpc_entry is not None
                    and tpc_entry.mtab_observed
                )
                if not _tpc_already:
                    _prompt_tokens_snapshot = list(prompt_tokens)
                    _sys_raw_for_mtab = sys_raw if 'sys_raw' in locals() else None

                    def _bg_mtab():
                        from app.innovation.mtab.tier_cache import get_tier_cache, TierEntry
                        import hashlib
                        tc = get_tier_cache()
                        tc.lookup_best(_prompt_tokens_snapshot)
                        h = hashlib.sha256(
                            ",".join(str(t) for t in _prompt_tokens_snapshot).encode("utf-8")
                        ).hexdigest()[:32]
                        tc.store(TierEntry(
                            prompt_hash=h, prompt_tokens=_prompt_tokens_snapshot,
                            layer_boundary=16, hidden_states=None,
                        ))
                        if _sys_raw_for_mtab:
                            get_tpc_cache().mark_mtab_observed(_sys_raw_for_mtab)

                    get_background_compiler().submit("mtab_observe", _bg_mtab)
        except Exception:
            logger.debug("MTAB observation hook failed", exc_info=True)

        # ── GENERATE via custom_generate (Innovation hot path) ─
        effective_max = plan.effective_max_output_tokens
        token_count = 0

        # ── RDMS: Resident Draft Model Speculative ──
        # Hybrid-main (Qwen3.5 GDN+FA) takes the save/restore speculative path
        # and only wins when accept rate is high enough. Narrative workloads
        # commonly sit around 20-30% accept, so RDMS stays OFF by default on
        # hybrid; opt in explicitly via speed_priority="high" or
        # plan.speed.use_speculative=True.
        use_rdms = False
        try:
            from app.innovation.rdms import is_rdms_available
            if is_rdms_available():
                from app.innovation.speculative.draft_speculative import _is_pure_transformer
                main_lm_probe = model.language_model if hasattr(model, 'language_model') else model
                main_inner_probe = main_lm_probe.model if hasattr(main_lm_probe, 'model') else main_lm_probe
                is_pure = _is_pure_transformer(main_inner_probe)
                use_rdms = is_pure or plan.speed.use_speculative
        except Exception:
            use_rdms = False

        # Legacy path: pre-attached draft + pure Transformer fast path
        use_legacy_spec = False
        if not use_rdms and self._draft_model is not None and plan.speed.use_speculative:
            from app.innovation.speculative.draft_speculative import _is_pure_transformer
            main_lm = model.language_model if hasattr(model, 'language_model') else model
            main_inner = main_lm.model if hasattr(main_lm, 'model') else main_lm
            use_legacy_spec = _is_pure_transformer(main_inner)

        if use_rdms:
            try:
                from app.innovation.rdms import rdms_speculative_generate
                _rdms_k = get_settings().innovation.rdms_num_draft
                # T2 CPP — per-request routing log moved to DEBUG.
                logger.debug("InnovationExecutor: using RDMS path (resident draft, k=%d)", _rdms_k)
                for chunk in rdms_speculative_generate(
                    main_model=model,
                    main_tokenizer=tokenizer,
                    prompt_tokens=prompt_tokens,
                    max_tokens=effective_max,
                    num_draft=_rdms_k,
                    temperature=0.0,
                ):
                    token_count += 1
                    yield chunk
                # Successful RDMS generation — done
                return
            except RuntimeError as e:
                # RDMS auto-disabled or unavailable — fall through to standard path
                logger.warning("RDMS path failed (%s) — falling back to standard", e)

        if use_legacy_spec:
            from app.innovation.speculative.draft_speculative import speculative_generate
            for chunk in speculative_generate(
                main_model=model,
                draft_model=self._draft_model,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                max_tokens=effective_max,
                num_draft=get_settings().innovation.rdms_num_draft,
                temperature=0.0,
            ):
                token_count += 1
                yield chunk
        else:
            # Per-request quality monitor (anomaly detection)
            q_monitor = QualityMonitor()

            gen_cfg = get_settings().generation
            effective_kv_bits = gen_cfg.kv_bits
            # Phase D FPQM — per-request override takes precedence.
            if plan.kv_precision_override == "bf16":
                effective_kv_bits = None  # full precision
                logger.info("FPQM: bf16 KV requested — full-precision path engaged")
            elif plan.kv_precision_override == "int4":
                effective_kv_bits = 4
            if effective_kv_bits is not None and ctrsp_manager and ctrsp_hash:
                logger.debug(
                    "CTRSP active: kv_bits=%s silently disabled (full-precision KV)",
                    effective_kv_bits,
                )

            # SSD predictor (per-request, opt-in via mode)
            ssd_predictor = None
            try:
                from app.core.config import get_settings as _gs
                feat = _gs().runtime.active_features()
                if feat.ssd_enabled:
                    from app.innovation.ssd import SkipPredictor
                    ssd_predictor = SkipPredictor(enabled=True)
            except Exception:
                ssd_predictor = None

            for chunk in generate_with_innovations(
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                max_tokens=effective_max,
                temperature=0.0,
                ctrsp_manager=ctrsp_manager,
                ctrsp_prompt_hash=ctrsp_hash,
                ctrsp_prefix_len=ctrsp_prefix_len,
                ctrsp_model_name=model_id,
                skip_controller=self._skip_controller,
                kv_bits=effective_kv_bits,
                lookahead_k=gen_cfg.lookahead_k,
                quality_monitor=q_monitor,
                ssd_predictor=ssd_predictor,
            ):
                token_count += 1
                yield chunk

        # ── Post-generation cleanup ───────────────────────────
        if self._long_gen_opt.is_active():
            self._long_gen_opt.deactivate()

        elapsed = time.perf_counter() - start_time
        ctrsp_status = "off"
        if ctrsp_manager and ctrsp_hash:
            cached = ctrsp_manager.get_cached_state(ctrsp_hash, model_id)
            if cached:
                ctrsp_status = f"cached ({cached.tokens_processed} tok)"
                self._stats["ctrsp_hits"] += 1
            else:
                ctrsp_status = "saved"
                self._stats["ctrsp_saves"] += 1

        logger.info(
            "InnovationExecutor: %d tokens in %.0fms, ctrsp=%s",
            token_count, elapsed * 1000, ctrsp_status,
        )

    def get_stats(self) -> dict:
        stats = dict(self._stats)
        if self._dpc_router:
            stats["dpc"] = self._dpc_router.get_stats()
        return stats
