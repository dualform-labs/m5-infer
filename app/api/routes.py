"""API routes — OpenAI-compatible endpoints with Innovation pipeline.

All requests flow through:
  Request → Planner (GeneratePlan) → InnovationExecutor → Response

The InnovationExecutor applies N1/X2/X4 hooks around the MLX generation loop.
"""

import asyncio
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatMessage,
    UsageInfo,
    ModelInfo,
    ModelListResponse,
    HealthResponse,
    ModelPullRequest,
    ModelPullResponse,
)
from app.engine.main_model_manager import MainModelManager
from app.engine.session_manager import SessionManager
from app.engine.memory_guard import MemoryGuard
from app.engine.cache_manager import CacheManager
from app.engine.supervisor import Supervisor
from app.engine.innovation_executor import InnovationExecutor
from app.engine.context_compressor import ContextCompressor
from app.engine.oirc import get_oirc, is_oirc_eligible, CachedChunk
from app.planner.request_planner import RequestPlanner
from app.backend.generation import build_prompt_tokens
from app.core.logging import get_logger, MetricsLogger

logger = get_logger(__name__)
metrics = MetricsLogger()

router = APIRouter()

# ── Adaptive Thinking rules ──────────────────────────────
_THINKING_ON_KEYWORDS = {"ultrathink", "think step by step", "think carefully", "reason through"}
_SIMPLE_PATTERNS = {
    "what is", "who is", "how many", "how much",
    "translate", "convert", "define", "list",
}


def _resolve_thinking(messages: list[dict], user_setting: bool) -> bool:
    """Decide whether to enable thinking mode.

    Rules (in priority order):
    1. "ultrathink" anywhere in user message → FORCE ON
    1b. **Needle retrieval detected** (long system context + short retrieval
        query) → FORCE ON. Rationale: Qwen3.5 の safety alignment は
        thinking OFF で retrieval 質問を拒否することがある (full_suite の
        "magic activation code" で発火、2026-04-19 確認)。thinking ON なら
        モデルは context から retrieve 可能。速度は retrieval 時のみ影響。
    2. User explicitly set enable_thinking=false → OFF
    3. Short question (< 30 chars, no complex keywords) → OFF
    4. Simple pattern match → OFF
    5. Default → ON
    """
    # Get the last user message
    last_user = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user = (msg.get("content") or "").strip()
            break

    if not last_user:
        return user_setting

    last_lower = last_user.lower()

    # Rule 1: "ultrathink" → always ON regardless of other settings
    if "ultrathink" in last_lower:
        return True

    # T10 — Respect explicit user intent. When the caller passed
    # enable_thinking=false we must not flip it ON via server-side
    # heuristics; doing so causes TTFT to include the full thinking
    # generation (2-10x observed latency). The needle-retrieval heuristic
    # below is advisory only when the user left thinking at its default.
    if not user_setting:
        return False

    # Rule 1b: Needle-retrieval heuristic — long context + short user query.
    # Only consulted when the caller did NOT explicitly set thinking=false.
    try:
        sys_chars = sum(
            len(m.get("content") or "") for m in messages if m.get("role") == "system"
        )
        if sys_chars > 3000 and len(last_user) < 200:
            question_like = (
                "?" in last_user
                or any(last_lower.startswith(w) for w in
                       ("what", "which", "where", "who", "when", "find", "extract",
                        "list", "name", "tell me"))
            )
            if question_like:
                logger.debug(
                    "_resolve_thinking: needle-retrieval heuristic → thinking ON "
                    "(sys=%d chars, user=%d chars)", sys_chars, len(last_user),
                )
                return True
    except Exception:
        pass

    # Rule 3: Very short simple questions → OFF
    if len(last_user) < 30 and "?" in last_user:
        # Check it's not a complex question
        complex_words = {"explain", "why", "how does", "analyze", "compare", "implement", "design", "debug"}
        if not any(w in last_lower for w in complex_words):
            return False

    # Rule 4: Simple pattern → OFF
    if any(last_lower.startswith(p) for p in _SIMPLE_PATTERNS):
        # Unless it's followed by complex context
        if len(last_user) < 80:
            return False

    # Rule 5: Default ON for complex/long questions
    return True

# Set by server lifespan via init_routes()
model_manager: MainModelManager | None = None
session_manager: SessionManager | None = None
planner: RequestPlanner | None = None
memory_guard: MemoryGuard | None = None
cache_manager: CacheManager | None = None
supervisor: Supervisor | None = None
executor: InnovationExecutor | None = None
context_compressor: ContextCompressor | None = None

# Serialize MLX model access. mlx_lm/MLX are not designed for concurrent
# forward passes on the same model — two in-flight generate_stream calls will
# interleave tokens and corrupt KV cache. Held across the entire generator.
_generation_lock: asyncio.Lock = asyncio.Lock()
# Exclusive lock for model swap / load; blocks while any generation is active.
_model_swap_lock: asyncio.Lock = asyncio.Lock()


def init_routes(
    mm: MainModelManager,
    sm: SessionManager,
    rp: RequestPlanner,
    mg: MemoryGuard,
    cm: CacheManager,
    sv: Supervisor,
    ie: InnovationExecutor,
    cc: ContextCompressor | None = None,
) -> None:
    global model_manager, session_manager, planner, memory_guard
    global cache_manager, supervisor, executor, context_compressor
    model_manager = mm
    session_manager = sm
    planner = rp
    memory_guard = mg
    cache_manager = cm
    supervisor = sv
    executor = ie
    context_compressor = cc


@router.get("/health")
async def health() -> HealthResponse:
    if model_manager is None or not model_manager.is_ready():
        return HealthResponse(status="not_ready")
    info = await model_manager.health_check()
    # Phase C observability
    from app.engine.mmrs_registry import get_mmrs_registry
    from app.engine.mrpb_scheduler import get_mrpb_scheduler
    info["mmrs"] = get_mmrs_registry().stats()
    info["mrpb"] = get_mrpb_scheduler().stats()
    # Phase D closing: MTAB observation hit-rate.
    try:
        from app.innovation.mtab.tier_cache import get_tier_cache
        info["mtab"] = get_tier_cache().stats()
    except Exception:
        info["mtab"] = None
    return HealthResponse(**info)


@router.get("/v1/models")
async def list_models() -> ModelListResponse:
    models = []
    if model_manager and model_manager.is_ready():
        model_id = model_manager.get_current_model_id()
        name = model_manager.get_backend().get_model_name()
        models.append(ModelInfo(id=model_id or name))
    return ModelListResponse(data=models)


def _hf_cache_size_bytes(repo_id: str) -> int:
    """On-disk size of a HF cache entry (sum of blobs/)."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:
        return 0
    from pathlib import Path
    folder = Path(HF_HUB_CACHE) / ("models--" + repo_id.replace("/", "--"))
    if not folder.is_dir():
        return 0
    total = 0
    blobs = folder / "blobs"
    if blobs.is_dir():
        for b in blobs.iterdir():
            try:
                total += b.stat().st_size
            except OSError:
                pass
    return total


def _classify_pull_error(exc: BaseException) -> tuple[str, str]:
    """Return (short_code, human_message) for a snapshot_download/load failure."""
    s = str(exc)
    sl = s.lower()
    if "429" in s or "rate" in sl and "limit" in sl:
        return (
            "rate_limited",
            f"HuggingFace rate-limited the download. Set HF_TOKEN to lift the limit: {s}",
        )
    if "no space" in sl or "disk" in sl and "full" in sl or "enospc" in sl:
        return ("disk_full", f"Not enough disk space to store the model: {s}")
    if "401" in s or "403" in s or "unauthorized" in sl or "forbidden" in sl:
        return (
            "auth",
            f"HuggingFace auth failed. Set HF_TOKEN if the repo is gated: {s}",
        )
    if "getaddrinfo" in sl or "network" in sl or "timed out" in sl or "timeout" in sl:
        return ("network", f"Network error contacting HuggingFace: {s}")
    if "metal" in sl and ("memory" in sl or "oom" in sl):
        return (
            "hardware_oom",
            f"Model too large for the available Metal memory: {s}. "
            f"Try a smaller variant or raise [memory] headroom in engine.toml.",
        )
    return ("error", s)


@router.post("/v1/models/pull")
async def pull_model(request: ModelPullRequest, http_request: Request):
    """Download and load a model from HuggingFace.

    Supports two modes:

    1. **Synchronous** (default, v1.1.0-compatible): blocks until download
       and load complete, then returns a single ``ModelPullResponse``.
    2. **Streaming** (v1.1.2+): set ``stream=true`` or send
       ``Accept: text/event-stream``. Returns newline-delimited JSON
       chunks with per-tick progress (``bytes_dl``, ``mbps``, ``eta_s``)
       during the download phase, then a final ``{"status": "..."}`` chunk
       on completion.

    In streaming mode the master lock is released during the HuggingFace
    network phase, so ``/health`` and chat against the currently-loaded
    model stay responsive while a download runs. The master lock is
    reacquired only for the MLX load / backend swap at the end.
    """
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    # Determine mode. Client can opt-in via body flag OR Accept header.
    wants_stream = bool(request.stream)
    try:
        accept = http_request.headers.get("accept", "")
        if "text/event-stream" in accept:
            wants_stream = True
    except Exception:
        pass

    repo = request.model

    if not wants_stream:
        # v1.1.0-compatible synchronous path
        async with _model_swap_lock:
            async with _generation_lock:
                try:
                    result = await model_manager.load_model(repo)
                    if executor is not None:
                        executor.set_backend(model_manager.get_backend())
                        executor.initialize_innovations()
                    return ModelPullResponse(
                        status=result["status"],
                        model=repo,
                        memory=result.get("memory"),
                    )
                except Exception as e:
                    logger.exception("Failed to pull model: %s", repo)
                    code, msg = _classify_pull_error(e)
                    raise HTTPException(
                        status_code=502,
                        detail={"error_code": code, "error": msg, "model": repo},
                    )

    # Streaming mode: download without master lock, load with master lock.
    async def event_stream():
        import json as _json
        import threading as _threading
        import time as _time

        done = _threading.Event()
        err: dict = {}

        def _worker():
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=repo)
            except Exception as e:
                code, msg = _classify_pull_error(e)
                err["code"] = code
                err["msg"] = msg
            finally:
                done.set()

        t0 = _time.perf_counter()
        last_size = _hf_cache_size_bytes(repo)
        last_ts = t0

        # Kick off download in a background thread; meanwhile we poll cache size.
        thr = _threading.Thread(target=_worker, name="m5-hf-pull", daemon=True)
        thr.start()
        yield _json.dumps({
            "phase": "download_start", "model": repo, "bytes_dl": last_size,
        }) + "\n"

        while not done.is_set():
            await asyncio.sleep(1.0)
            now = _time.perf_counter()
            size = _hf_cache_size_bytes(repo)
            dt = now - last_ts
            mbps = ((size - last_size) / (1 << 20)) / dt if dt > 0 else 0
            yield _json.dumps({
                "phase": "downloading",
                "model": repo,
                "bytes_dl": size,
                "mbps": round(mbps, 2),
                "elapsed_s": round(now - t0, 1),
            }) + "\n"
            last_size = size
            last_ts = now

        thr.join(timeout=1.0)

        if err:
            yield _json.dumps({
                "phase": "error",
                "error_code": err["code"],
                "error": err["msg"],
                "model": repo,
            }) + "\n"
            return

        # Download complete — now take the master lock for the MLX load phase.
        yield _json.dumps({
            "phase": "load_start",
            "model": repo,
            "bytes_dl": _hf_cache_size_bytes(repo),
        }) + "\n"

        try:
            async with _model_swap_lock:
                async with _generation_lock:
                    result = await model_manager.load_model(repo)
                    if executor is not None:
                        executor.set_backend(model_manager.get_backend())
                        executor.initialize_innovations()
        except Exception as e:
            logger.exception("Failed to load model after download: %s", repo)
            code, msg = _classify_pull_error(e)
            yield _json.dumps({
                "phase": "error", "error_code": code, "error": msg, "model": repo,
            }) + "\n"
            return

        yield _json.dumps({
            "phase": "success",
            "status": result["status"],
            "model": repo,
            "memory": result.get("memory"),
        }) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    # If model field specifies a HF repo, auto-load it.
    # Guard with swap lock + generation lock so we don't pull the model out
    # from under an in-flight request, and on failure preserve the current
    # model rather than leaving the server in a half-loaded state.
    if request.model and request.model != "default" and "/" in request.model:
        current = model_manager.get_current_model_id()
        if current != request.model:
            async with _model_swap_lock:
                async with _generation_lock:
                    current = model_manager.get_current_model_id()
                    if current != request.model:
                        logger.info("Auto-loading model from request: %s", request.model)
                        try:
                            await model_manager.load_model(request.model)
                            if executor is not None:
                                executor.set_backend(model_manager.get_backend())
                                executor.initialize_innovations()
                        except Exception as e:
                            logger.exception("Auto-load failed: %s", request.model)
                            raise HTTPException(
                                status_code=502,
                                detail=f"Auto-load failed: {e}",
                            )

    if not model_manager.is_ready():
        raise HTTPException(status_code=503, detail="Model not ready")

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    backend = model_manager.get_backend()

    # ── Session ───────────────────────────────────────────────
    session = session_manager.get_or_create(request.session_id)

    # ── Prompt construction ───────────────────────────────────
    messages_dicts = [m.model_dump(exclude_none=True) for m in request.messages]

    # ── Context compression (old history) ────────────────────
    if context_compressor and backend:
        try:
            if context_compressor.should_compress(messages_dicts, backend.get_tokenizer()):
                messages_dicts = context_compressor.compress(messages_dicts, backend.get_tokenizer())
        except Exception:
            logger.warning("Context compression failed, using original messages")

    # ── Adaptive Thinking ─────────────────────────────────
    enable_thinking = _resolve_thinking(messages_dicts, request.enable_thinking)

    prompt_tokens = build_prompt_tokens(
        messages_dicts, backend.get_tokenizer(), request.tools,
        enable_thinking=enable_thinking,
    )

    # ── Planner ── creates GeneratePlan that CONTROLS execution ─
    num_tools = len(request.tools) if request.tools else 0
    plan = planner.plan(
        messages=messages_dicts,
        max_tokens=request.max_tokens,
        prompt_token_count=len(prompt_tokens),
        has_image=False,
        speed_priority_hint=request.speed_priority,
        prefer_long_generation=request.prefer_long_generation,
        prefer_long_context=request.prefer_long_context,
        disable_speculative=request.disable_speculative,
        force_cache_preserve=request.force_cache_preserve,
        session_is_continuation=request.session_id is not None,
        num_tool_schemas=num_tools,
        kv_precision=request.kv_precision,
        prefer_quality=request.prefer_quality,
    )

    if plan.should_reject:
        raise HTTPException(
            status_code=429, detail=plan.reject_reason or "Request rejected"
        )

    # T2 CPP — route-level plan summary moved to DEBUG (was INFO).
    # Redundant with the InnovationExecutor summary logged AFTER generation
    # completes; keeping the post-hoc summary is enough for observability.
    logger.debug(
        "Plan: model=%s speed=%s ctrsp=%s fold=%s max_out=%d",
        plan.routing.selected_model,
        plan.speed.speed_priority.value,
        plan.enable_ctrsp,
        plan.enable_context_fold,
        plan.effective_max_output_tokens,
    )

    # ── Execute via InnovationExecutor (plan decisions are APPLIED here) ─
    start_time = time.perf_counter()

    # T14-OIRC — Opt-In Response Cache.
    # Consulted ONLY when the caller supplies both `idempotency_key` and a
    # positive `cache_ttl_ms`. Without both, every request takes the normal
    # generation path — which is what an agent issuing "re-check current
    # state" expects. Stochastic and tool-using requests short-circuit before
    # we ever touch the cache.
    _oirc = None
    _oirc_model_id = ""
    _oirc_cached = None
    if is_oirc_eligible(
        idempotency_key=request.idempotency_key,
        cache_ttl_ms=request.cache_ttl_ms,
        temperature=0.0,
        top_p=1.0,
        max_tokens=request.max_tokens,
        tools=request.tools,
    ):
        _oirc = get_oirc()
        _oirc_model_id = model_manager.get_current_model_id()
        _oirc_cached = _oirc.get(
            _oirc_model_id,
            request.idempotency_key,
            prompt_tokens,
            request.max_tokens,
            enable_thinking,
        )

    if request.stream:
        if _oirc_cached is not None:
            return StreamingResponse(
                _stream_from_oirc(request_id, created, request.model, _oirc_cached),
                media_type="text/event-stream",
            )
        return StreamingResponse(
            _stream_via_executor(
                request_id, created, plan, messages_dicts, prompt_tokens,
                request, session,
                oirc=_oirc,
                oirc_model_id=_oirc_model_id,
                oirc_idempotency_key=request.idempotency_key,
                oirc_max_tokens=request.max_tokens,
                oirc_enable_thinking=enable_thinking,
                oirc_ttl_ms=request.cache_ttl_ms,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming: collect all chunks from executor (serialized)
    text_parts = []
    finish_reason = None
    gen_tokens = 0
    gen_tps = 0.0
    peak_mem = 0.0
    first_token_time = None

    gen = None
    async with _generation_lock:
        try:
            if supervisor:
                supervisor.report_generation_start()
            gen = executor.execute(
                plan=plan,
                messages=messages_dicts,
                prompt_tokens=prompt_tokens,
                tools=request.tools,
                request_id=request_id,
                session_id=session.session_id,
            )
            for chunk in gen:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                if supervisor and supervisor.stall_detected():
                    logger.error("Supervisor reports stall, aborting generation")
                    break
                text_parts.append(chunk.text)
                finish_reason = chunk.finish_reason
                gen_tokens = chunk.generation_tokens
                gen_tps = chunk.generation_tps
                peak_mem = chunk.peak_memory_gb
        except Exception as e:
            if supervisor:
                supervisor.report_generation_failure()
            logger.exception("Non-stream generation failed")
            raise HTTPException(status_code=500, detail=f"Generation error: {e}")
        finally:
            if gen is not None:
                try:
                    gen.close()
                except Exception:
                    pass

    full_text = "".join(text_parts)
    total_time = time.perf_counter() - start_time
    ttft = (first_token_time - start_time) * 1000 if first_token_time else 0

    try:
        session_manager.append_message(session.session_id, "assistant", full_text)
    except Exception:
        logger.warning("Failed to save response to session %s", session.session_id)

    metrics.log_request(
        request_id=request_id,
        session_id=session.session_id,
        model=plan.routing.selected_model,
        prompt_tokens=len(prompt_tokens),
        output_tokens=gen_tokens,
        ttft_ms=ttft,
        decode_tok_per_sec=gen_tps,
        total_latency_ms=total_time * 1000,
        peak_memory_gb=peak_mem,
        prefix_cache_hit=plan.enable_prefix_cache,
        speculative_used=plan.speed.use_speculative,
        finish_reason=finish_reason or "stop",
    )

    if supervisor:
        supervisor.report_generation_activity()

    return ChatCompletionResponse(
        id=request_id,
        created=created,
        model=request.model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=full_text),
                finish_reason=finish_reason or "stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=len(prompt_tokens),
            completion_tokens=gen_tokens,
            total_tokens=len(prompt_tokens) + gen_tokens,
        ),
    )


async def _stream_from_oirc(request_id, created, model_name, cached):
    """T14-OIRC — replay a previously cached response for an opt-in idempotent
    call. Every chunk is exactly what the model emitted on the original run.
    """
    first = True
    for cc in cached.chunks:
        if not cc.text and cc.finish_reason is None:
            continue
        if first and cc.text:
            sse_chunk = ChatCompletionChunk(
                id=request_id, created=created, model=model_name,
                choices=[ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(role="assistant", content=cc.text),
                )],
            )
            first = False
        else:
            sse_chunk = ChatCompletionChunk(
                id=request_id, created=created, model=model_name,
                choices=[ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(content=cc.text),
                )],
            )
        yield f"data: {sse_chunk.model_dump_json()}\n\n"
    final_reason = "stop"
    for cc in reversed(cached.chunks):
        if cc.finish_reason:
            final_reason = cc.finish_reason
            break
    final = ChatCompletionChunk(
        id=request_id, created=created, model=model_name,
        choices=[ChatCompletionChunkChoice(
            delta=ChatCompletionChunkDelta(),
            finish_reason=final_reason,
        )],
    )
    yield f"data: {final.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_via_executor(
    request_id, created, plan, messages_dicts, prompt_tokens, request, session,
    oirc=None, oirc_model_id="", oirc_idempotency_key="",
    oirc_max_tokens=0, oirc_enable_thinking=False, oirc_ttl_ms=0,
):
    """SSE streaming via InnovationExecutor.

    Guarantees:
      * Only one generation runs at a time (holds _generation_lock for the
        lifetime of the underlying generator).
      * On client disconnect / CancelledError, the underlying generator is
        .close()'d so MLX releases its wired_limit and cache state promptly.
      * Session is updated only after successful generation — partial writes
        on abort are discarded to avoid corrupting next turn's context.
      * T14-OIRC — when the caller supplied an idempotency_key + cache_ttl_ms,
        the full SSE chunk list is committed to the cache after a clean
        completion, bounded by ttl_ms.
    """
    text_parts: list[str] = []
    first = True
    last_finish_reason: str | None = None
    completed = False
    gen = None
    _oirc_chunks: list[CachedChunk] = []

    await _generation_lock.acquire()
    try:
        if supervisor:
            supervisor.report_generation_start()
        gen = executor.execute(
            plan=plan,
            messages=messages_dicts,
            prompt_tokens=prompt_tokens,
            tools=request.tools,
            request_id=request_id,
            session_id=session.session_id,
        )
        try:
            for chunk in gen:
                if supervisor and supervisor.stall_detected():
                    logger.error("Supervisor reports stall, aborting SSE")
                    break
                if chunk.finish_reason is not None:
                    last_finish_reason = chunk.finish_reason

                if first:
                    sse_chunk = ChatCompletionChunk(
                        id=request_id, created=created, model=request.model,
                        choices=[ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(
                                role="assistant", content=chunk.text,
                            ),
                        )],
                    )
                    first = False
                else:
                    sse_chunk = ChatCompletionChunk(
                        id=request_id, created=created, model=request.model,
                        choices=[ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(content=chunk.text),
                        )],
                    )

                text_parts.append(chunk.text)
                if oirc is not None:
                    _oirc_chunks.append(CachedChunk(
                        text=chunk.text,
                        finish_reason=chunk.finish_reason,
                        generation_tokens=getattr(chunk, "generation_tokens", 0),
                    ))
                yield f"data: {sse_chunk.model_dump_json()}\n\n"
            completed = True
        except asyncio.CancelledError:
            logger.info("SSE client disconnected — aborting generation")
            raise
        except Exception:
            logger.exception("Error during streaming generation")
            if supervisor:
                supervisor.report_generation_failure()

        if completed and not first:
            final = ChatCompletionChunk(
                id=request_id, created=created, model=request.model,
                choices=[ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(),
                    finish_reason=last_finish_reason or "stop",
                )],
            )
            yield f"data: {final.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

            # T14-OIRC — commit the replay record now that generation finished
            # cleanly. The caller must have opted in; the cache is never
            # populated by default.
            if oirc is not None and _oirc_chunks and oirc_ttl_ms:
                _oirc_chunks[-1] = CachedChunk(
                    text=_oirc_chunks[-1].text,
                    finish_reason=last_finish_reason or "stop",
                    generation_tokens=_oirc_chunks[-1].generation_tokens,
                )
                try:
                    oirc.put(
                        oirc_model_id,
                        oirc_idempotency_key,
                        prompt_tokens,
                        oirc_max_tokens,
                        oirc_enable_thinking,
                        _oirc_chunks,
                        oirc_ttl_ms,
                    )
                except Exception:
                    logger.debug("OIRC put failed", exc_info=True)
    finally:
        if gen is not None:
            try:
                gen.close()
            except Exception:
                logger.debug("Generator close raised", exc_info=True)
        _generation_lock.release()
        if supervisor:
            supervisor.report_generation_activity()

        if completed:
            full_text = "".join(text_parts)
            try:
                session_manager.append_message(session.session_id, "assistant", full_text)
            except Exception:
                logger.warning("Failed to save response to session %s", session.session_id)
