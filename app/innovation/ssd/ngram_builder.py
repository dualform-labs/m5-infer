"""SSD n-gram builder — offline corpus → table extraction.

Usage (offline, run once per (model, tokenizer) combination):

    python3 -m app.innovation.ssd.ngram_builder \
        --model mlx-community/Qwen3.5-9B-MLX-4bit \
        --corpus path/to/corpus.txt \
        --output state/ssd_table.json \
        --min-confidence 0.95 \
        --ngram-size 5

The script:
  1. Tokenizes the corpus
  2. Counts n-gram → next-token frequencies
  3. For each n-gram with conditional P(next | ngram) >= min_confidence,
     records it in the table.
  4. Writes JSON for runtime loading.

Phase A: stub — full implementation deferred. Provides CLI interface
and documents the format. Empty table = SSD is no-op (safe).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def build_ngram_table(
    corpus_text: str,
    tokenizer,
    ngram_size: int = 5,
    min_confidence: float = 0.95,
    min_occurrences: int = 50,
) -> dict:
    """Scan corpus, return SSD table dict ready to JSON-serialize."""
    tokens = tokenizer.encode(corpus_text)
    # Count (ngram → next_token) frequencies
    ngram_followers: dict[tuple, Counter] = defaultdict(Counter)
    for n in range(2, ngram_size + 1):
        for i in range(len(tokens) - n):
            ctx = tuple(tokens[i : i + n])
            nxt = tokens[i + n]
            ngram_followers[ctx][nxt] += 1

    entries = []
    for ctx, follower_counts in ngram_followers.items():
        total = sum(follower_counts.values())
        if total < min_occurrences:
            continue
        most_common, count = follower_counts.most_common(1)[0]
        conf = count / total
        if conf >= min_confidence:
            entries.append({
                "context": list(ctx),
                "next": most_common,
                "confidence": round(conf, 3),
            })

    return {
        "ngram_size": ngram_size,
        "min_confidence": min_confidence,
        "entries": entries,
    }


def main():
    parser = argparse.ArgumentParser(description="Build SSD n-gram table")
    parser.add_argument("--model", required=True, help="Tokenizer model HF path")
    parser.add_argument("--corpus", required=True, help="Path to corpus text file")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--min-confidence", type=float, default=0.95)
    parser.add_argument("--ngram-size", type=int, default=5)
    parser.add_argument("--min-occurrences", type=int, default=50)
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model}...")
    from mlx_lm import load
    _, tok = load(args.model)

    print(f"Reading corpus {args.corpus}...")
    text = Path(args.corpus).read_text(encoding="utf-8")
    print(f"  corpus chars: {len(text)}")

    print(f"Building n-gram table (n={args.ngram_size}, min_conf={args.min_confidence})...")
    table = build_ngram_table(
        text, tok,
        ngram_size=args.ngram_size,
        min_confidence=args.min_confidence,
        min_occurrences=args.min_occurrences,
    )
    print(f"  found {len(table['entries'])} deterministic n-grams")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(table, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
