"""API routes — OpenAI-compatible endpoints with Innovation pipeline.

All requests flow through:
  Request → Planner (GeneratePlan) → InnovationExecutor → Response

The InnovationExecutor applies N1/X2/X4 hooks around the MLX generation loop.
"""

import asyncio
import time
import uuid

from fastapi import APIRouter, HTTPException
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

    # Rule 1b: Needle-retrieval heuristic — long context + short user query.
    # Detection:
    #   - sum(system msg chars) > 3000 (≈ 750+ tokens), AND
    #   - user message < 200 chars, AND
    #   - user message looks like a question (contains "?" or starts with what/which/where)
    # If matched, force thinking ON to avoid safety-refusal on long contexts.
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
                logger.info(
                    "_resolve_thinking: needle-retrieval heuristic → thinking ON "
                    "(sys=%d chars, user=%d chars)", sys_chars, len(last_user),
                )
                return True
    except Exception:
        pass

    # Rule 2: User explicitly disabled thinking
    if not user_setting:
        return False

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


@router.post("/v1/models/pull")
async def pull_model(request: ModelPullRequest) -> ModelPullResponse:
    """Download and load a model from HuggingFace.

    Any MLX-compatible model can be loaded by its HuggingFace repo ID.
    The swap is serialized — concurrent generation requests block until
    the swap completes (or fails).
    """
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    # Take both locks: block new generations and wait for in-flight ones.
    async with _model_swap_lock:
        async with _generation_lock:
            try:
                result = await model_manager.load_model(request.model)
                if executor is not None:
                    executor.set_backend(model_manager.get_backend())
                    executor.initialize_innovations()
                return ModelPullResponse(
                    status=result["status"],
                    model=request.model,
                    memory=result.get("memory"),
                )
            except Exception as e:
                logger.exception("Failed to pull model: %s", request.model)
                # Return an actual HTTP error so clients don't think the swap
                # succeeded when the model is now broken/absent.
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to load model {request.model}: {e}",
                )


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

    logger.info(
        "Plan: model=%s speed=%s ctrsp=%s fold=%s max_out=%d",
        plan.routing.selected_model,
        plan.speed.speed_priority.value,
        plan.enable_ctrsp,
        plan.enable_context_fold,
        plan.effective_max_output_tokens,
    )

    # ── Execute via InnovationExecutor (plan decisions are APPLIED here) ─
    start_time = time.perf_counter()

    if request.stream:
        return StreamingResponse(
            _stream_via_executor(
                request_id, created, plan, messages_dicts, prompt_tokens,
                request, session,
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


async def _stream_via_executor(
    request_id, created, plan, messages_dicts, prompt_tokens, request, session
):
    """SSE streaming via InnovationExecutor.

    Guarantees:
      * Only one generation runs at a time (holds _generation_lock for the
        lifetime of the underlying generator).
      * On client disconnect / CancelledError, the underlying generator is
        .close()'d so MLX releases its wired_limit and cache state promptly.
      * Session is updated only after successful generation — partial writes
        on abort are discarded to avoid corrupting next turn's context.
    """
    text_parts: list[str] = []
    first = True
    last_finish_reason: str | None = None
    completed = False
    gen = None

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
