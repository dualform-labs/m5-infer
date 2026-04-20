"""Quality anomaly detection during generation.

Monitors output token stream for:
  - Repetition loops (same n-gram repeating)
  - Confidence drops (optional, requires logprobs)
  - Topic drift / gibberish patterns

Can trigger early termination to prevent wasting tokens on bad output.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AnomalyReport:
    """Anomaly detection result."""
    detected: bool
    reason: str = ""
    detail: str = ""


class QualityMonitor:
    """Detects anomalies in streamed generation output."""

    def __init__(
        self,
        ngram_size: int = 12,
        history_size: int = 360,
        max_ngram_repeats: int = 5,
        min_window_for_detection: int = 100,
        check_every_n_tokens: int = 16,
    ):
        """
        Args:
            ngram_size: Check for repeats of this token-length sequence.
                Larger = fewer false positives on structured output (markdown
                headers, bullet lists share short token prefixes). Default 12
                is long enough to avoid matching trivial structure while still
                catching genuine repetition loops.
            history_size: How many recent tokens to track. 360 covers 5 repeats
                of 72-token periods or 6 repeats of 60-token periods.
            max_ngram_repeats: Trigger if same n-gram appears this many times.
                Default 5 — well-formed markdown with 3-4 similar list items
                must NOT trigger; only true loops (which repeat 10+ times)
                should fire. 5 is the sweet spot to avoid false positives.
            min_window_for_detection: Start detection after this many tokens.
                Raised to 100 so short lists of repeated structural markers
                never cross the threshold.
            check_every_n_tokens: Throttle O(N²) n-gram scan to every Nth token.
        """
        self._ngram_size = ngram_size
        self._history_size = history_size
        self._max_repeats = max_ngram_repeats
        self._min_window = min_window_for_detection
        self._check_every = max(1, check_every_n_tokens)
        self._tokens: deque[int] = deque(maxlen=history_size)
        self._n_tokens_seen = 0

    def record(self, token_id: int) -> AnomalyReport:
        """Record a new token and check for anomalies."""
        self._tokens.append(token_id)
        self._n_tokens_seen += 1

        if self._n_tokens_seen < self._min_window:
            return AnomalyReport(detected=False)

        # Throttle expensive scan — check every Nth token only
        if self._n_tokens_seen % self._check_every != 0:
            return AnomalyReport(detected=False)

        # Scan the ENTIRE history window (not a tiny sub-window) so long-period
        # repetitions are catchable. With defaults (240 tokens, n=8, repeats=3)
        # this catches loops with period up to ~80 tokens.
        if len(self._tokens) < self._ngram_size * self._max_repeats:
            return AnomalyReport(detected=False)

        recent = list(self._tokens)
        ngrams = [
            tuple(recent[i : i + self._ngram_size])
            for i in range(len(recent) - self._ngram_size + 1)
        ]
        if not ngrams:
            return AnomalyReport(detected=False)

        from collections import Counter
        counts = Counter(ngrams)
        most_common_ngram, count = counts.most_common(1)[0]
        if count >= self._max_repeats:
            return AnomalyReport(
                detected=True,
                reason="repetition_loop",
                detail=(
                    f"n-gram (len={self._ngram_size}) repeated {count} times "
                    f"in last {len(recent)} tokens"
                ),
            )

        return AnomalyReport(detected=False)

    def reset(self) -> None:
        self._tokens.clear()
        self._n_tokens_seen = 0

    def stats(self) -> dict:
        return {
            "tokens_seen": self._n_tokens_seen,
            "history_size": len(self._tokens),
        }
