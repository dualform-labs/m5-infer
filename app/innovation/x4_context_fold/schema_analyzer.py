"""X4: Context Folding — Tool schema structural analysis.

Detects common token patterns across tool schemas to enable KV sharing.
Agent system prompts often contain 20-50 tool schemas with identical JSON structure.
This module finds the common template and per-schema deltas.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from app.core.logging import get_logger

logger = get_logger(__name__)

@dataclass
class SchemaTemplate:
    """A shared template extracted from multiple tool schemas."""
    template_tokens: list[int]      # Common token sequence
    delta_positions: list[int]      # Positions where schemas differ
    delta_tokens: list[list[int]]   # Per-schema tokens at delta positions
    schema_count: int               # Number of schemas that share this template
    fold_ratio: float               # Fraction of tokens that are shared (0-1)

@dataclass
class FoldAnalysis:
    """Result of analyzing tool schemas for folding opportunities."""
    templates: list[SchemaTemplate]
    total_tokens: int               # Total tokens across all schemas
    shared_tokens: int              # Tokens that can be shared
    unique_tokens: int              # Tokens unique to individual schemas
    fold_ratio: float               # Overall fold ratio
    worth_folding: bool             # True if folding would save significant compute


def analyze_tool_schemas(
    tools: list[dict],
    tokenizer,
    min_schemas_for_folding: int = 5,
    min_fold_ratio: float = 0.2,
) -> FoldAnalysis:
    """Analyze tool schemas to find common patterns for KV sharing.

    Each tool is typically: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
    The JSON structure is highly repetitive across tools.
    """
    if not tools or len(tools) < min_schemas_for_folding:
        return FoldAnalysis(
            templates=[], total_tokens=0, shared_tokens=0,
            unique_tokens=0, fold_ratio=0.0, worth_folding=False,
        )

    # Tokenize each tool schema
    import json
    schema_token_lists = []
    for tool in tools:
        tool_str = json.dumps(tool, ensure_ascii=False)
        tokens = tokenizer.encode(tool_str)
        schema_token_lists.append(tokens)

    total_tokens = sum(len(t) for t in schema_token_lists)

    # Find common prefix and suffix across all schemas
    common_prefix_len = _find_common_prefix_length(schema_token_lists)
    common_suffix_len = _find_common_suffix_length(schema_token_lists)

    # Prevent prefix+suffix overlap (they must not exceed shortest schema)
    min_schema_len = min(len(t) for t in schema_token_lists)
    if common_prefix_len + common_suffix_len > min_schema_len:
        # Prefer prefix over suffix when they overlap
        common_suffix_len = max(0, min_schema_len - common_prefix_len)

    # Calculate fold ratio
    if not schema_token_lists:
        return FoldAnalysis(
            templates=[], total_tokens=0, shared_tokens=0,
            unique_tokens=0, fold_ratio=0.0, worth_folding=False,
        )

    avg_len = total_tokens / len(schema_token_lists)
    shared_per_schema = common_prefix_len + common_suffix_len
    fold_ratio = shared_per_schema / avg_len if avg_len > 0 else 0.0

    # Build template
    if fold_ratio >= min_fold_ratio and len(schema_token_lists) >= min_schemas_for_folding:
        template_tokens = schema_token_lists[0][:common_prefix_len]
        if common_suffix_len > 0:
            template_tokens += schema_token_lists[0][-common_suffix_len:]

        # Extract deltas (the unique middle portion of each schema)
        deltas = []
        for tokens in schema_token_lists:
            if common_suffix_len > 0:
                delta = tokens[common_prefix_len:-common_suffix_len]
            else:
                delta = tokens[common_prefix_len:]
            deltas.append(delta)

        # Delta positions are between prefix and suffix
        delta_positions = list(range(common_prefix_len, common_prefix_len + max(len(d) for d in deltas)))

        template = SchemaTemplate(
            template_tokens=template_tokens,
            delta_positions=delta_positions,
            delta_tokens=deltas,
            schema_count=len(schema_token_lists),
            fold_ratio=fold_ratio,
        )

        shared_tokens = shared_per_schema * len(schema_token_lists)
        unique_tokens = total_tokens - shared_tokens

        logger.info(
            "Context Folding: %d schemas, fold_ratio=%.2f, shared=%d, unique=%d tokens",
            len(schema_token_lists), fold_ratio, shared_tokens, unique_tokens,
        )

        return FoldAnalysis(
            templates=[template],
            total_tokens=total_tokens,
            shared_tokens=shared_tokens,
            unique_tokens=unique_tokens,
            fold_ratio=fold_ratio,
            worth_folding=True,
        )
    else:
        return FoldAnalysis(
            templates=[], total_tokens=total_tokens, shared_tokens=0,
            unique_tokens=total_tokens, fold_ratio=fold_ratio, worth_folding=False,
        )


def _find_common_prefix_length(token_lists: list[list[int]]) -> int:
    """Find the length of the common prefix across all token lists."""
    if not token_lists:
        return 0
    min_len = min(len(t) for t in token_lists)
    prefix_len = 0
    for i in range(min_len):
        if all(t[i] == token_lists[0][i] for t in token_lists):
            prefix_len = i + 1
        else:
            break
    return prefix_len


def _find_common_suffix_length(token_lists: list[list[int]]) -> int:
    """Find the length of the common suffix across all token lists."""
    if not token_lists:
        return 0
    min_len = min(len(t) for t in token_lists)
    suffix_len = 0
    for i in range(1, min_len + 1):
        if all(t[-i] == token_lists[0][-i] for t in token_lists):
            suffix_len = i
        else:
            break
    return suffix_len
