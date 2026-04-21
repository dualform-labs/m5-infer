from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.routes import router, init_routes, metrics as route_metrics
from app.engine.main_model_manager import MainModelManager
from app.engine.session_manager import SessionManager
from app.engine.memory_guard import MemoryGuard
from app.engine.cache_manager import CacheManager
from app.engine.supervisor import Supervisor
from app.planner.request_planner import RequestPlanner
from app.engine.innovation_executor import InnovationExecutor
from app.engine.context_compressor import ContextCompressor
from app.engine.sub_model_controller import SubModelController
from app.storage.sqlite_store import SQLiteStore
from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.core.thread_priority import apply_to_current_process

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    setup_logging()
    logger.info("m5-infer starting up...")

    settings = get_settings()

    # ── Hardware detection + auto-tune (v1.0) ─────────────
    # Non-fatal: if detection fails, we log a warning and continue
    # with whatever is in engine.toml. Apple Silicon is strongly
    # recommended; M1+ supported, M5 series recommended.
    try:
        from app.core.hardware_profile import detect_hardware
        from app.core.auto_tune import compute_overrides, apply_overrides

        chip = detect_hardware()
        logger.info("Hardware: %s", chip.summary())
        for w in chip.warnings:
            logger.warning("Hardware: %s", w)

        if not chip.is_apple_silicon:
            logger.error(
                "m5-infer requires Apple Silicon (M1 or newer). "
                "Current system: %s. Continuing but performance and "
                "stability are not guaranteed.", chip.chip_name or "unknown",
            )

        if getattr(settings.engine, "auto_tune", True):
            overrides = compute_overrides(chip)
            apply_overrides(settings, overrides)

            # Apply Metal memory limit recommendation (safe — clamps to
            # a fraction of total memory, OS keeps its headroom)
            if overrides.wired_limit_mb:
                try:
                    import mlx.core as mx
                    if hasattr(mx, "metal") and hasattr(mx.metal, "set_memory_limit"):
                        mx.metal.set_memory_limit(
                            int(overrides.wired_limit_mb * 1024 * 1024),
                        )
                        logger.info(
                            "Metal memory_limit set to %d MB (total %d GB - headroom)",
                            overrides.wired_limit_mb, int(chip.memory_gb),
                        )
                except Exception as _exc:
                    logger.debug("Metal memory_limit skipped: %s", _exc)
        else:
            logger.info("Hardware: auto_tune disabled; using engine.toml values verbatim.")

        # Opportunity logs — point out capabilities we don't yet exploit
        if chip.is_apple_silicon and chip.ane_tops > 0:
            logger.info(
                "Hardware: Apple Neural Engine detected (%d cores, ~%.0f TOPS) — "
                "currently NOT used by m5-infer inference engine. "
                "ANE offload for prefill / speculative draft is on the roadmap.",
                chip.ane_cores, chip.ane_tops,
            )
        # MLX version check for Metal 4 tensor op availability
        try:
            import mlx
            mlx_v = getattr(mlx, "__version__", "unknown")
            if chip.supports_metal_4:
                logger.info(
                    "Hardware: macOS %s + mlx %s — Metal 4 fused ops expected "
                    "(scaled_dot_product_attention, rope, rms_norm).",
                    chip.macos_version, mlx_v,
                )
        except Exception:
            pass
    except Exception:
        logger.exception("Hardware detection or auto-tune failed (non-fatal)")

    rt = settings.runtime
    feat = rt.active_features()
    logger.info(
        "runtime mode = %s | features: ctrsp_lru=%d, draft=%s, "
        "x2_shadow=%s, mtab=%s, pst=%s, ssd=%s",
        rt.memory_mode, feat.ctrsp_lru_size,
        feat.draft_model_path or "(disabled)",
        feat.x2_shadow_enabled, feat.mtab_enabled,
        feat.pst_enabled, feat.ssd_enabled,
    )

    # Set Super core priority for inference thread
    apply_to_current_process()

    mm = MainModelManager()
    sm = SessionManager()
    mg = MemoryGuard()
    cm = CacheManager()
    rp = RequestPlanner(memory_guard=mg)
    sv = Supervisor(memory_guard=mg, session_manager=sm)

    await mm.startup()

    # ── RDMS: Resident draft model (if configured for active mode) ──
    if feat.draft_model_path:
        from app.innovation.rdms.draft_loader import get_draft_loader
        draft_loader = get_draft_loader()
        ok = draft_loader.load(feat.draft_model_path)
        if ok:
            from app.innovation.rdms.hybrid_speculative import check_tokenizer_compat
            main_tok = mm.get_backend().get_tokenizer()
            _, draft_tok = draft_loader.get()
            check_tokenizer_compat(main_tok, draft_tok)
        else:
            logger.warning(
                "RDMS: draft load failed for %s — RDMS path will be skipped",
                feat.draft_model_path,
            )

    # InnovationExecutor: bridges plan decisions to actual inference
    ie = InnovationExecutor(backend=mm.get_backend(), cache_manager=cm)
    ie.initialize_innovations()

    # Context compressor for long conversation history
    cc = ContextCompressor()

    # Sub model controller for dynamic model loading
    smc = SubModelController(memory_guard=mg)

    # ── MMRS: activate resident registry only when total budget allows
    # a second model next to the 9B main + shadow + KV runtime (~13 GB used).
    # Threshold 32 GB chosen so 24 GB M5 base stays on the load/unload path.
    from app.engine.mmrs_registry import get_mmrs_registry
    mem_cfg = settings.memory
    total_budget = getattr(mem_cfg, "total_budget_gb", 24.0)
    if total_budget >= 32.0:
        get_mmrs_registry().activate(total_resident_gb_budget=total_budget - 12.0)
        logger.info("MMRS: enabled (total=%.0f GB)", total_budget)
    else:
        logger.info("MMRS: gated off (total=%.0f GB < 32 GB, load/unload mode)",
                    total_budget)

    # ── MRPB scheduler: single-class serialize by default
    from app.engine.mrpb_scheduler import get_mrpb_scheduler
    _ = get_mrpb_scheduler()  # instantiate so routes pick it up lazily

    # SQLite store for persistent metrics
    sqlite_store = SQLiteStore()
    route_metrics.set_store(sqlite_store)

    # Wire supervisor remediation hooks so memory RED triggers actual cleanup
    from app.innovation.n1_ctrsp.session_integration import get_ctrsp_manager
    sv.set_remediation_hooks(
        ctrsp_manager=get_ctrsp_manager(),
        cache_manager=cm,
    )
    await sv.start()
    init_routes(mm, sm, rp, mg, cm, sv, ie, cc=cc)

    app.state.model_manager = mm
    app.state.session_manager = sm
    app.state.memory_guard = mg
    app.state.cache_manager = cm
    app.state.planner = rp
    app.state.supervisor = sv
    app.state.executor = ie
    app.state.context_compressor = cc
    app.state.sub_model_controller = smc
    app.state.sqlite_store = sqlite_store

    logger.info("M5 Inference Engine ready")
    yield

    logger.info("M5 Inference Engine shutting down...")
    smc.unload_all()
    await sv.stop()
    sqlite_store.close()
    await mm.shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="M5 MLX Inference Engine",
        version=__import__("app").__version__,
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


def main():
    import uvicorn
    settings = get_settings()
    app = create_app()
    uvicorn.run(app, host=settings.server.host, port=settings.server.port)


if __name__ == "__main__":
    main()
