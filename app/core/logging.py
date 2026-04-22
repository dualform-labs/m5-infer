"""Structured JSON logging for the M5 MLX Inference Engine."""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any


# ---------------------------------------------------------------------------
# Request context (per-request, stored in a ContextVar)
# ---------------------------------------------------------------------------

@dataclass
class RequestContext:
    """Carries per-request identifiers through async call stacks."""

    request_id: str
    session_id: str | None = None
    model: str | None = None


_request_ctx: ContextVar[RequestContext | None] = ContextVar(
    "m5_request_context", default=None
)


def get_request_context() -> RequestContext | None:
    """Return the current RequestContext, or None if not set."""
    return _request_ctx.get()


def set_request_context(ctx: RequestContext) -> None:
    """Bind a RequestContext for the current task / coroutine."""
    _request_ctx.set(ctx)


def clear_request_context() -> None:
    """Remove the current RequestContext."""
    _request_ctx.set(None)


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects.

    Output schema (guaranteed keys):
        {"timestamp", "level", "module", "message", ...extra_fields}

    Any extra attributes attached to the LogRecord are merged into the
    top-level object.  The active RequestContext (if any) is included
    automatically.
    """

    # Keys that belong to the standard LogRecord and should NOT be forwarded
    # as extra fields.
    _BUILTIN_ATTRS: frozenset[str] = frozenset(
        {
            "args",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "taskName",
            "thread",
            "threadName",
        }
    )

    # JST (UTC+9) for local display
    _JST = timezone(timedelta(hours=9))

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        local_time = datetime.fromtimestamp(record.created, tz=self._JST)
        payload: dict[str, Any] = {
            "timestamp": local_time.strftime("%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }

        # Merge RequestContext when available.
        ctx = _request_ctx.get()
        if ctx is not None:
            payload.update(
                {k: v for k, v in asdict(ctx).items() if v is not None}
            )

        # Merge any extra fields the caller attached to the record.
        for key, value in record.__dict__.items():
            if key not in self._BUILTIN_ATTRS and key not in payload:
                payload[key] = value

        # Include exception info when present.
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            payload["exception"] = record.exc_text
        if record.stack_info:
            payload["stack_info"] = record.stack_info

        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with the JSON formatter.

    Calling this more than once is safe — it replaces the existing
    handlers on the root logger rather than stacking duplicates.

    Third-party HTTP / download loggers (``huggingface_hub._client``,
    ``httpx``, ``urllib3``) are forced to WARNING so their per-request
    chatter does not drown out engine events at the default INFO level.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove any pre-existing handlers to avoid duplicate output.
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)

    # v1.1.2: silence the noisy HuggingFace / HTTP clients at INFO level.
    # Set them to WARNING so only real problems surface, keeping the engine's
    # own INFO-level events readable in `m5-infer start` stdout tails.
    for noisy in (
        "huggingface_hub._client",
        "huggingface_hub._http",
        "huggingface_hub.file_download",
        "huggingface_hub.utils._http",
        "httpx",
        "httpcore",
        "urllib3",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (thin wrapper around ``logging.getLogger``)."""
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Metrics logger
# ---------------------------------------------------------------------------

class MetricsLogger:
    """Structured logger for inference-level metrics.

    All events are emitted at INFO through the ``m5infer.metrics`` logger,
    making it easy to route them to a dedicated sink.
    """

    def __init__(self, sqlite_store=None) -> None:
        self._logger = logging.getLogger("m5infer.metrics")
        self._sqlite_store = sqlite_store

    def set_store(self, store) -> None:
        """Attach a SQLiteStore for persistent metrics."""
        self._sqlite_store = store

    def log_request(
        self,
        *,
        request_id: str,
        session_id: str | None = None,
        model: str | None = None,
        prompt_tokens: int,
        output_tokens: int,
        ttft_ms: float,
        decode_tok_per_sec: float,
        total_latency_ms: float,
        peak_memory_gb: float,
        prefix_cache_hit: bool,
        speculative_used: bool,
        finish_reason: str,
        **extra: Any,
    ) -> None:
        """Log a single inference request with its associated metrics."""
        metrics: dict[str, Any] = {
            "event": "inference_request",
            "request_id": request_id,
            "session_id": session_id,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "ttft_ms": round(ttft_ms, 1),
            "decode_tok_per_sec": round(decode_tok_per_sec, 1),
            "total_latency_ms": round(total_latency_ms, 1),
            "peak_memory_gb": round(peak_memory_gb, 2),
            "prefix_cache_hit": prefix_cache_hit,
            "speculative_used": speculative_used,
            "finish_reason": finish_reason,
        }
        if extra:
            metrics.update(extra)

        # Human-readable speed summary line
        speed_summary = (
            f"TTFT={ttft_ms:.0f}ms | "
            f"{decode_tok_per_sec:.1f} tok/s | "
            f"{output_tokens} tokens in {total_latency_ms:.0f}ms | "
            f"mem={peak_memory_gb:.2f}GB"
        )

        self._logger.info(
            speed_summary,
            extra=metrics,
        )

        if self._sqlite_store:
            try:
                self._sqlite_store.log_request(**{
                    "request_id": request_id,
                    "session_id": session_id,
                    "model": model,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "ttft_ms": ttft_ms,
                    "decode_tok_per_sec": decode_tok_per_sec,
                    "total_latency_ms": total_latency_ms,
                    "peak_memory_gb": peak_memory_gb,
                    "prefix_cache_hit": prefix_cache_hit,
                    "speculative_used": speculative_used,
                    "finish_reason": finish_reason,
                    **extra,
                })
            except Exception:
                pass  # Don't fail on metrics persistence
