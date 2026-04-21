from pydantic import BaseModel, Field
from typing import Any, Literal


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, ge=1, le=262144)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    stream: bool = False
    stop: list[str] | None = None
    tools: list[dict] | None = None
    repetition_penalty: float = Field(1.0, ge=0.0, le=2.0)
    # m5-infer extensions
    session_id: str | None = None
    speed_priority: Literal["normal", "high", "max_long_generation", "vision_fast"] | None = None
    prefer_long_context: bool = False
    prefer_long_generation: bool = False
    disable_speculative: bool = False
    force_cache_preserve: bool = False
    enable_thinking: bool = True  # Set to false to disable <think> blocks (much faster)
    # Phase D FPQM — per-request KV precision override.
    # "int4" (default) = quantized KV cache, "bf16" = full-precision (higher
    # quality, ~2× KV memory). None = inherit from engine.toml.
    kv_precision: Literal["int4", "bf16"] | None = None
    # Phase D FPQM — convenience flag that maps to kv_precision="bf16" + lossless
    # flags (disables speculative, forces greedy verify). Takes precedence.
    prefer_quality: bool = False
    # T14-OIRC — opt-in response cache. Both fields must be provided for the
    # server to consider a cached replay. Without them, every call is a fresh
    # model execution (agent re-check semantics preserved by default).
    idempotency_key: str | None = Field(default=None, max_length=128)
    cache_ttl_ms: int | None = Field(default=None, ge=0, le=60_000)


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class ChatCompletionChunkDelta(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "m5-infer"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    model: str | None = None
    memory: dict | None = None
    mmrs: dict | None = None   # resident registry stats
    mrpb: dict | None = None   # request-class scheduler stats
    mtab: dict | None = None   # tier cache hit stats


class ModelPullRequest(BaseModel):
    """Request to download and load a model from HuggingFace."""
    model: str  # HuggingFace repo ID, e.g. "mlx-community/Llama-3.2-3B-Instruct-4bit"

class ModelPullResponse(BaseModel):
    status: str  # "loaded", "already_loaded", "error"
    model: str
    memory: dict | None = None
    error: str | None = None
