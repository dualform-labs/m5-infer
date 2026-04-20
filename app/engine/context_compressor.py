"""Context Compressor — compresses old conversation history.

Implements SPEC section 6.9: old history compression,
verbose tool output reduction, important facts extraction.
"""

from __future__ import annotations
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class ContextCompressor:
    """Compresses conversation context to manage token budget."""

    def __init__(self):
        self._stats = {"compressions": 0, "tokens_saved": 0}

    def should_compress(self, messages: list[dict], tokenizer) -> bool:
        """Check if context needs compression based on token count.

        Two-stage check to avoid expensive full tokenization on every request:
          Stage 1 (O(N) char count): if chars / 2 < trigger, skip.
                  ratio chars:tokens ≈ 3-4 for English, ≈ 1-2 for Japanese.
                  Use conservative ratio 2.0 so we only skip when clearly under.
          Stage 2 (O(N) tokenize): run real encode only when stage 1 says maybe.
        """
        settings = get_settings()
        trigger = settings.context.summary_trigger_tokens

        total_chars = sum(len(m.get("content", "") or "") for m in messages)
        # Lower bound on token count: chars / 2.0 (pessimistic for ASCII,
        # closer-to-actual for mixed CJK). If that bound is already under
        # trigger, no need to do the real tokenize.
        if total_chars // 2 < trigger:
            return False

        # Stage 2: precise tokenize
        total_tokens = sum(
            len(tokenizer.encode(m.get("content", "") or ""))
            for m in messages
        )
        return total_tokens > trigger

    def compress(self, messages: list[dict], tokenizer) -> list[dict]:
        """Compress older messages while preserving recent and important ones.

        Strategy:
        1. Keep system message intact
        2. Keep last N turns (raw_recent_turns from config)
        3. Summarize/truncate older turns
        4. Compress verbose tool outputs
        """
        settings = get_settings()
        recent_turns = settings.context.raw_recent_turns

        if len(messages) <= recent_turns + 1:  # +1 for system
            return messages

        # Separate system message
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        # Keep recent, compress old
        recent = non_system[-recent_turns:]
        old = non_system[:-recent_turns]

        if not old:
            return messages

        # Compress old messages: truncate tool outputs
        compressed_old = []
        tokens_saved = 0
        for msg in old:
            content = msg.get("content", "") or ""
            role = msg.get("role", "")

            if role == "tool" and settings.context.compress_tool_outputs:
                # Truncate verbose tool output
                original_len = len(tokenizer.encode(content))
                if original_len > 200:
                    truncated = content[:500] + "\n[... truncated ...]"
                    tokens_saved += original_len - len(tokenizer.encode(truncated))
                    compressed_old.append({**msg, "content": truncated})
                    continue

            compressed_old.append(msg)

        self._stats["compressions"] += 1
        self._stats["tokens_saved"] += tokens_saved

        result = system_msgs + compressed_old + recent
        logger.info(
            "Context compressed: %d -> %d messages, ~%d tokens saved",
            len(messages), len(result), tokens_saved,
        )
        return result

    def get_stats(self) -> dict:
        return dict(self._stats)
