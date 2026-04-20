"""N1 CTRSP integration with session and cache managers."""

from __future__ import annotations
from app.innovation.n1_ctrsp.state_persistence import CTRSPManager
from app.core.logging import get_logger

logger = get_logger(__name__)

# Module-level singleton
_ctrsp_manager: CTRSPManager | None = None


def get_ctrsp_manager() -> CTRSPManager:
    """Get the singleton CTRSPManager with optional disk persistence.

    max_cached_states is read from the active runtime mode
    (engine.toml [runtime.<mode>] ctrsp_lru_size). Default 32 for moderate.
    """
    global _ctrsp_manager
    if _ctrsp_manager is None:
        from pathlib import Path
        from app.core.config import get_settings
        # Mode-aware LRU size (scales with memory tier).
        feat = get_settings().runtime.active_features()
        lru_size = feat.ctrsp_lru_size
        persist_dir = Path("state/ctrsp")
        _ctrsp_manager = CTRSPManager(
            max_cached_states=lru_size,
            persist_dir=str(persist_dir),
        )
        logger.info(
            "M-CTRSP: initialized with LRU size = %d (mode=%s)",
            lru_size, get_settings().runtime.memory_mode,
        )
    return _ctrsp_manager


def extract_system_prompt_tokens(messages: list[dict], tokenizer) -> list[int] | None:
    """Extract the tokenized system-prompt prefix as the model will see it.

    Many chat templates (Qwen3.5, Llama 3) reject a system-only message,
    so we render the system with TWO different dummy-user continuations and
    return their longest common prefix. That prefix is exactly the system
    portion — independent of any user payload — and therefore matches the
    leading slice of the full templated prompt regardless of what the real
    user message is.

    Returns None if there's no system message or the tokenizer can't render.
    """
    sys_msg = None
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if content:
                sys_msg = {"role": "system", "content": content}
                break
    if sys_msg is None:
        return None

    if not hasattr(tokenizer, "apply_chat_template"):
        return None

    def _render(user_text: str) -> list[int] | None:
        try:
            out = tokenizer.apply_chat_template(
                [sys_msg, {"role": "user", "content": user_text}],
                tokenize=True, add_generation_prompt=False,
            )
            if hasattr(out, "tolist"):
                out = out.tolist()
            return list(out)
        except Exception:
            return None

    # Two disjoint dummy strings so the common prefix is guaranteed to end
    # at the system boundary (any tokens from dummy user content differ).
    a = _render("A")
    b = _render("Z")
    if not a or not b:
        logger.warning("CTRSP tokenizer can't render chat template — CTRSP disabled")
        return None

    # Common prefix
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    if i == 0:
        return None
    return a[:i]


def should_use_ctrsp(
    session_is_continuation: bool,
    system_prompt_tokens: list[int] | None,
    model_name: str,
) -> bool:
    """Determine if CTRSP should be used for this request."""
    if not session_is_continuation:
        return False
    if system_prompt_tokens is None:
        return False

    manager = get_ctrsp_manager()
    prompt_hash = manager.compute_prompt_hash(system_prompt_tokens)
    cached = manager.get_cached_state(prompt_hash, model_name)
    return cached is not None
