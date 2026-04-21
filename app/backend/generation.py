"""Prompt construction and token utilities for OpenAI-compatible chat messages."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_prompt(
    messages: list[dict],
    tokenizer,
    tools: list[dict] | None = None,
) -> str:
    """Build a formatted prompt string from OpenAI-format chat messages.

    Uses ``tokenizer.apply_chat_template`` when available, falling back to
    simple role/content concatenation otherwise.

    Args:
        messages: List of ``{"role": ..., "content": ...}`` dicts.
        tokenizer: A tokenizer object (e.g. from ``transformers`` or ``mlx-lm``).
        tools: Optional list of tool/function definitions to pass to the template.

    Returns:
        The fully formatted prompt string ready for generation.
    """
    if not messages:
        return ""

    # Prefer the tokenizer's own chat template
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            kwargs: dict = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if tools:
                kwargs["tools"] = tools
            result = tokenizer.apply_chat_template(messages, **kwargs)
            if isinstance(result, str):
                return result
            # Some tokenizers return a list of token-ids even with tokenize=False
            # when misconfigured; fall through to the fallback in that case.
            logger.warning(
                "apply_chat_template returned %s instead of str; using fallback",
                type(result).__name__,
            )
        except Exception:
            logger.warning(
                "apply_chat_template failed; falling back to simple concatenation",
                exc_info=True,
            )

    # Fallback: simple concatenation
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "") or ""
        parts.append(f"<|{role}|>\n{content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def count_tokens(text: str | None, tokenizer) -> int:
    """Return the number of tokens in *text*.

    Args:
        text: The string to tokenize.  ``None`` or empty string returns 0.
        tokenizer: A tokenizer with an ``encode`` method.

    Returns:
        Token count as an integer.
    """
    if not text:
        return 0
    try:
        ids = tokenizer.encode(text)
        return len(ids)
    except Exception:
        logger.warning("tokenizer.encode failed; returning 0", exc_info=True)
        return 0


# T12 FPTC — Full-Prompt Token Cache.
# apply_chat_template + tokenize of a 12K-token prompt costs 30-100 ms every
# request. For the common warm-reuse pattern (same messages + tools sent
# back-to-back) we cache the tokenized result keyed by a fast bytes-level
# fingerprint of the full request shape. Bounded LRU prevents unbounded
# memory use. Correctness: cache key encodes every input that would affect
# the template output, so a hit is byte-equivalent to a re-tokenize.
import hashlib as _hashlib
from collections import OrderedDict as _OrderedDict
from threading import Lock as _Lock

_FPTC_MAX = 32
_fptc_cache: "_OrderedDict[str, list[int]]" = _OrderedDict()
_fptc_lock = _Lock()


def _fptc_key(messages: list[dict], tools: list[dict] | None, enable_thinking: bool) -> str:
    h = _hashlib.sha256()
    for m in messages:
        h.update(b"\x00")
        h.update((m.get("role") or "").encode("utf-8", errors="replace"))
        h.update(b"\x01")
        c = m.get("content")
        if isinstance(c, str):
            h.update(c.encode("utf-8", errors="replace"))
        elif isinstance(c, list):
            for part in c:
                h.update(str(part).encode("utf-8", errors="replace"))
    h.update(b"\x02tools=")
    if tools:
        for t in tools:
            h.update(str(t).encode("utf-8", errors="replace"))
            h.update(b"\x03")
    h.update(b"\x04thinking=")
    h.update(b"1" if enable_thinking else b"0")
    return h.hexdigest()


def build_prompt_tokens(
    messages: list[dict],
    tokenizer,
    tools: list[dict] | None = None,
    enable_thinking: bool = True,
) -> list[int]:
    """Build token IDs from OpenAI-format chat messages.

    Tries ``tokenizer.apply_chat_template`` with ``tokenize=True`` first,
    then falls back to encoding the string produced by :func:`build_prompt`.
    T12 — Hot-path cache short-circuits identical requests.

    Args:
        messages: List of ``{"role": ..., "content": ...}`` dicts.
        tokenizer: A tokenizer object.
        tools: Optional list of tool/function definitions.
        enable_thinking: If False, passes enable_thinking=False to chat template
            to suppress <think> blocks (dramatically faster for Qwen3.5).

    Returns:
        List of token IDs.
    """
    if not messages:
        return []

    # T12 FPTC — full-prompt LRU cache
    key = _fptc_key(messages, tools, enable_thinking)
    with _fptc_lock:
        hit = _fptc_cache.get(key)
        if hit is not None:
            _fptc_cache.move_to_end(key)
            return list(hit)  # defensive copy

    result_tokens: list[int] | None = None
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            kwargs: dict = {
                "tokenize": True,
                "add_generation_prompt": True,
            }
            if tools:
                kwargs["tools"] = tools
            if not enable_thinking:
                kwargs["enable_thinking"] = False
            result = tokenizer.apply_chat_template(messages, **kwargs)
            if isinstance(result, list):
                result_tokens = result
        except Exception:
            logger.warning(
                "apply_chat_template (tokenize=True) failed; using fallback",
                exc_info=True,
            )

    if result_tokens is None:
        # Fallback: build the string prompt and encode it
        prompt_str = build_prompt(messages, tokenizer, tools)
        try:
            result_tokens = list(tokenizer.encode(prompt_str))
        except Exception:
            logger.warning("tokenizer.encode failed; returning empty list", exc_info=True)
            return []

    with _fptc_lock:
        _fptc_cache[key] = list(result_tokens)
        _fptc_cache.move_to_end(key)
        while len(_fptc_cache) > _FPTC_MAX:
            _fptc_cache.popitem(last=False)

    return result_tokens


def estimate_output_tokens(
    messages: list[dict],
    tokenizer,
    max_tokens: int,
) -> tuple[int, int]:
    """Estimate prompt token count and effective max output tokens.

    Args:
        messages: List of ``{"role": ..., "content": ...}`` dicts.
        tokenizer: A tokenizer object.
        max_tokens: Caller-requested maximum output tokens.

    Returns:
        A ``(prompt_token_count, effective_max_output_tokens)`` tuple.
        Since the model's absolute context limit is not available here,
        ``effective_max_output_tokens`` is simply *max_tokens*.
    """
    token_ids = build_prompt_tokens(messages, tokenizer)
    prompt_count = len(token_ids)
    return prompt_count, max_tokens
