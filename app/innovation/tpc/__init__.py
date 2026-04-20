"""TPC — Token Prefix Compiler.

Novel innovation (2026-04-18) designed to win tool_heavy workloads where
mlx_lm's simple prompt_cache currently competes favorably with v3's rich
(but expensive) CTRSP + MTAB + redundancy-analysis pipeline.

Key ideas:
  1. Raw-bytes (UTF-8) hash of system content enables O(1) hit detection
     BEFORE tokenization. A hit short-circuits the full
     extract_system_prompt_tokens + SHA-of-token-list pipeline.
  2. Heavyweight analysis (redundancy scan, MTAB full-prompt observation)
     moves OFF the critical path to a background worker.
  3. Compiled artifacts are cached across sessions by content (not session_id),
     so a tool-heavy API that keeps reconnecting still benefits.

This module is additive: it augments CTRSP/MTAB with a stateless prelude,
not replaces them.
"""

from app.innovation.tpc.fast_lookup import (
    get_tpc_cache,
    TPCCache,
    TPCEntry,
)
from app.innovation.tpc.background_compiler import (
    get_background_compiler,
    BackgroundCompiler,
)

__all__ = [
    "get_tpc_cache",
    "TPCCache",
    "TPCEntry",
    "get_background_compiler",
    "BackgroundCompiler",
]
