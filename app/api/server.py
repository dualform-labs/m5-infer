from contextlib import asynccontextmanager
from pathlib import Path

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

            # Metal wired-memory limit — priority order (highest first):
            #   1. $M5_INFER_METAL_LIMIT_GB env var
            #   2. [memory] metal_limit_gb in engine.toml
            #   3. $M5_INFER_METAL_HEADROOM_GB env var
            #   4. [memory] headroom_gb in engine.toml
            #   5. auto_tune per-tier default (already in overrides.wired_limit_mb)
            import os as _os
            from app.core.auto_tune import _wired_limit_from_memory as _recompute

            abs_limit = None
            source = "auto_tune per-tier default"
            _env_abs = _os.getenv("M5_INFER_METAL_LIMIT_GB")
            if _env_abs:
                try:
                    abs_limit = float(_env_abs)
                    source = "$M5_INFER_METAL_LIMIT_GB env"
                except ValueError:
                    logger.warning("Invalid $M5_INFER_METAL_LIMIT_GB=%r (ignored)", _env_abs)
            elif settings.memory.metal_limit_gb is not None:
                abs_limit = float(settings.memory.metal_limit_gb)
                source = "engine.toml [memory] metal_limit_gb"

            headroom = None
            if abs_limit is None:
                _env_hr = _os.getenv("M5_INFER_METAL_HEADROOM_GB")
                if _env_hr:
                    try:
                        headroom = float(_env_hr)
                        source = "$M5_INFER_METAL_HEADROOM_GB env"
                    except ValueError:
                        logger.warning("Invalid $M5_INFER_METAL_HEADROOM_GB=%r (ignored)", _env_hr)
                elif settings.memory.headroom_gb is not None:
                    headroom = float(settings.memory.headroom_gb)
                    source = "engine.toml [memory] headroom_gb"

            if abs_limit is not None or headroom is not None:
                overrides.wired_limit_mb = _recompute(
                    chip.memory_gb,
                    headroom_override_gb=headroom,
                    absolute_limit_gb=abs_limit,
                )

            if overrides.wired_limit_mb:
                try:
                    import mlx.core as mx
                    if hasattr(mx, "metal") and hasattr(mx.metal, "set_memory_limit"):
                        mx.metal.set_memory_limit(
                            int(overrides.wired_limit_mb * 1024 * 1024),
                        )
                        limit_gb = overrides.wired_limit_mb / 1024.0
                        logger.info(
                            "Metal memory_limit set to %d MB (%.1f GB on %d GB total) — source: %s",
                            overrides.wired_limit_mb, limit_gb, int(chip.memory_gb), source,
                        )
                        # Warn on aggressive configuration
                        effective_headroom = chip.memory_gb - limit_gb
                        if effective_headroom < 3.0:
                            logger.warning(
                                "Metal headroom is only %.1f GB — macOS may swap under "
                                "memory pressure; inference tok/s may drop. Consider "
                                "closing other apps or loosening the limit.",
                                effective_headroom,
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

    # First-run: if the configured main model isn't in the HF cache, warn the
    # user that startup will block on a ~GB download. Best-effort — any error
    # here is non-fatal, we just skip the hint.
    try:
        from app.core.model_registry import get_registry
        _profile = get_registry().get_main()
        _repo = getattr(_profile, "path", None) or ""
        if "/" in _repo:  # HF repo id form "org/name"
            from huggingface_hub.constants import HF_HUB_CACHE
            _cache_dir = Path(HF_HUB_CACHE) / ("models--" + _repo.replace("/", "--"))
            _has_snapshot = _cache_dir.exists() and any(_cache_dir.glob("snapshots/*/config.json"))
            if not _has_snapshot:
                print(
                    f"\n[m5-infer] First run detected for {_repo}.\n"
                    f"           Downloading model from HuggingFace — this can take several minutes.\n"
                    f"           Progress appears on stderr below. Set HF_TOKEN to avoid rate limits.\n",
                    flush=True,
                )
    except Exception as _firstrun_err:
        logger.debug("first-run check skipped: %s", _firstrun_err)

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

    # Human-friendly single-line ready banner (goes to stdout so it's always
    # visible even when log level is WARN). Detailed state remains in logs.
    try:
        from app import __version__ as _ver
        from app.core.paths import data_root, find_config
        _cfg_path, _cfg_text = find_config("engine.toml")
        _cfg_shown = str(_cfg_path) if _cfg_path else "bundled defaults"
        _mem = mm.get_backend().get_memory_usage()
        print(
            f"\n[Ready] m5-infer v{_ver}  "
            f"http://{settings.server.host}:{settings.server.port}  |  "
            f"model: {mm.get_current_model_id() or settings.model.main_path or 'auto'}  |  "
            f"config: {_cfg_shown}  |  "
            f"data: {data_root()}  |  "
            f"resident: {_mem.get('active_gb', 0):.1f} GB\n",
            flush=True,
        )
    except Exception as _banner_err:  # never block startup on banner
        logger.debug("ready-banner skipped: %s", _banner_err)

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


def _cmd_init(target_dir: str | None = None) -> int:
    """Write bundled default config files to ``./configs/`` (or override dir)."""
    import os
    from importlib.resources import files as _res_files
    from pathlib import Path

    dst_dir = Path(target_dir).expanduser() if target_dir else (Path.cwd() / "configs")
    dst_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    for name in ("engine.toml", "models.toml"):
        try:
            src = _res_files("app").joinpath("_defaults", name)
        except (FileNotFoundError, ModuleNotFoundError):
            print(f"  ! bundled default for {name} not found (is m5-infer installed correctly?)")
            continue
        dst = dst_dir / name
        if dst.exists():
            print(f"  = {dst} already exists (skipped)")
            continue
        dst.write_bytes(src.read_bytes())
        print(f"  + {dst}")
        created += 1

    if created:
        print(f"\nCreated {created} config file(s) in {dst_dir}")
        print("Edit them to tune m5-infer, then run `m5-infer start`.")
    else:
        print(f"\nNo files written. {dst_dir} already has the config files.")
    return 0


def _cmd_start(args) -> int:
    """Run the foreground server, applying any CLI overrides."""
    import os
    import sys

    if args.config:
        os.environ["M5_INFER_CONFIG"] = str(Path(args.config).expanduser().resolve())

    try:
        settings = get_settings()
    except FileNotFoundError as e:
        print(f"[m5-infer] Configuration error:\n{e}", file=sys.stderr)
        print("\nHint: run `m5-infer init` to create a local ./configs/engine.toml.",
              file=sys.stderr)
        return 1

    host = args.host or settings.server.host
    port = args.port or settings.server.port

    import uvicorn
    try:
        app = create_app()
        uvicorn.run(app, host=host, port=port)
    except OSError as e:
        if "Address already in use" in str(e) or getattr(e, "errno", None) == 48:
            print(
                f"[m5-infer] Port {port} is already in use.\n"
                f"  Try: m5-infer start --port 11437\n"
                f"  Or edit [server] port in your engine.toml.",
                file=sys.stderr,
            )
            return 1
        raise
    return 0


def main():
    """Console entry point (``m5-infer``)."""
    import argparse
    from pathlib import Path
    from app import __version__

    parser = argparse.ArgumentParser(
        prog="m5-infer",
        description="MLX inference engine for Apple Silicon (OpenAI-compatible).",
    )
    parser.add_argument(
        "--version", action="version", version=f"m5-infer {__version__}",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to engine.toml (file) or configs/ (directory).",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Override server port (default from engine.toml).",
    )
    parser.add_argument(
        "--host", type=str, default=None,
        help="Override server host (default from engine.toml).",
    )

    sub = parser.add_subparsers(dest="command")
    sub.add_parser(
        "start",
        help="Start the server in the foreground (default if no command given).",
    )
    init_sub = sub.add_parser(
        "init",
        help="Create ./configs/engine.toml and models.toml from bundled defaults.",
    )
    init_sub.add_argument(
        "--dir", type=str, default=None,
        help="Target directory (default: ./configs).",
    )

    # v1.1.2 subcommands: client-side tooling that talks to a running server.
    chat_sub = sub.add_parser(
        "chat",
        help="Interactive chat CLI (requires a running server).",
    )
    chat_sub.add_argument(
        "model", nargs="?", default=None,
        help="Optional HuggingFace repo id to load before chatting.",
    )

    pull_sub = sub.add_parser(
        "pull",
        help="Download a model from HuggingFace through the server.",
    )
    pull_sub.add_argument("model", help="HuggingFace repo id (e.g. mlx-community/Qwen3.5-9B-MLX-4bit).")

    bench_sub = sub.add_parser(
        "bench",
        help="Quick decode smoke benchmark (tok/s + TTFT).",
    )
    bench_sub.add_argument("--full", action="store_true", help="Run more iterations (slower, tighter median).")
    bench_sub.add_argument("--quick", action="store_true", help="Run fewer iterations (default).")

    sub.add_parser("stop", help="Stop the m5-infer server listening on the configured port.")
    sub.add_parser("status", help="Print server health / loaded model / memory / cache stats.")

    cache_sub = sub.add_parser("cache", help="Inspect or purge engine caches (CTRSP / OIRC / TPC).")
    cache_action = cache_sub.add_subparsers(dest="cache_cmd")
    cache_action.add_parser("list", help="Show current cache sizes.")
    clear_cmd = cache_action.add_parser("clear", help="Purge caches.")
    clear_cmd.add_argument("--ctrsp", action="store_true", help="Clear disk CTRSP snapshots.")
    clear_cmd.add_argument("--oirc",  action="store_true", help="Clear in-memory OIRC (requires server restart).")
    clear_cmd.add_argument("--tpc",   action="store_true", help="Clear in-memory TPC (requires server restart).")
    clear_cmd.add_argument("--all",   action="store_true", help="Clear everything we can.")

    models_sub = sub.add_parser("models", help="List models loaded on the server or present in the HF cache.")
    models_sub.add_argument("--loaded", action="store_true", help="Show only the currently-loaded model.")
    models_sub.add_argument("--cached", action="store_true", help="Show only models in the HF cache.")

    args = parser.parse_args()

    # Dispatch
    if args.command == "init":
        return _cmd_init(target_dir=args.dir)
    if args.command == "chat":
        from app.cli.chat import run as chat_run
        return chat_run(args) or 0
    if args.command == "pull":
        from app.cli.pull import run as pull_run
        return pull_run(args) or 0
    if args.command == "bench":
        from app.cli.bench import run as bench_run
        return bench_run(args) or 0
    if args.command == "stop":
        from app.cli.status import run_stop
        return run_stop(args)
    if args.command == "status":
        from app.cli.status import run_status
        return run_status(args)
    if args.command == "cache":
        from app.cli.cache import run_list, run_clear
        if args.cache_cmd == "clear":
            return run_clear(args)
        return run_list(args)  # default to list if no sub
    if args.command == "models":
        from app.cli.models import run as models_run
        return models_run(args)

    # Default: start the server (args.command is None or "start")
    return _cmd_start(args)


if __name__ == "__main__":
    raise SystemExit(main() or 0)
