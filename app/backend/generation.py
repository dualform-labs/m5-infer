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


def build_prompt_tokens(
    messages: list[dict],
    tokenizer,
    tools: list[dict] | None = None,
    enable_thinking: bool = True,
) -> list[int]:
    """Build token IDs from OpenAI-format chat messages.

    Tries ``tokenizer.apply_chat_template`` with ``tokenize=True`` first,
    then falls back to encoding the string produced by :func:`build_prompt`.

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
                return result
        except Exception:
            logger.warning(
                "apply_chat_template (tokenize=True) failed; using fallback",
                exc_info=True,
            )

    # Fallback: build the string prompt and encode it
    prompt_str = build_prompt(messages, tokenizer, tools)
    try:
        return tokenizer.encode(prompt_str)
    except Exception:
        logger.warning("tokenizer.encode failed; returning empty list", exc_info=True)
        return []


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
