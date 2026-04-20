"""MTAB Prefix Matcher — find longest common prefix between two token lists."""

from __future__ import annotations


def find_longest_prefix_match(a: list[int], b: list[int]) -> int:
    """Return the length of the longest common prefix of token sequences a and b."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def find_best_match_in_set(
    query: list[int],
    candidates: list[list[int]],
    min_match: int = 256,
) -> tuple[int, int]:
    """Find which candidate has the longest prefix match with query.

    Args:
        query: input prompt tokens.
        candidates: list of cached prompt token lists.
        min_match: minimum match length to consider (avoid trivial matches).

    Returns:
        (best_index, best_length). best_index is -1 if no candidate ≥ min_match.
    """
    best_idx = -1
    best_len = 0
    for i, cand in enumerate(candidates):
        m = find_longest_prefix_match(query, cand)
        if m >= min_match and m > best_len:
            best_len = m
            best_idx = i
    return best_idx, best_len
