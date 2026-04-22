"""`m5-infer bench` — quick decode smoke benchmark.

Sends a short fixed prompt three times and reports median decode tokens-per-
second plus the warm and cold first-token latencies. Intended as a "did my
install actually install?" check, not a launch-quality benchmark (that work
lives in the internal bench harness that isn't shipped with the wheel).
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.error
import urllib.request

from app.cli._http import (
    BOLD,
    CYAN,
    DIM,
    GRAY,
    GREEN,
    NC,
    YELLOW,
    die_if_server_down,
    server_url,
)


PROMPT_QUICK = [
    {
        "role": "user",
        "content": "Count from 1 to 50 in English, separated by commas. No commentary.",
    },
]


def _measure_one(payload: dict) -> tuple[float, float, int]:
    """Run one streaming decode; return (ttft_s, elapsed_s, tok_count)."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{server_url()}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    ttft: float | None = None
    tok_count = 0
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload_line = line[5:].strip()
            if payload_line == "[DONE]":
                break
            try:
                data = json.loads(payload_line)
            except json.JSONDecodeError:
                continue
            choices = data.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if delta.get("content") or choices[0].get("text"):
                if ttft is None:
                    ttft = time.perf_counter() - start
                tok_count += 1
    elapsed = time.perf_counter() - start
    return ttft or elapsed, elapsed, tok_count


def run(args) -> int:
    die_if_server_down()

    max_tokens = 128 if getattr(args, "full", False) else 64
    warmup_runs = 1
    measure_runs = 5 if getattr(args, "full", False) else 3

    print(f"{BOLD}m5-infer bench{NC}  {GRAY}{server_url()}{NC}")
    print(
        f"  prompt: 'Count from 1 to 50' · max_tokens={max_tokens} · "
        f"warmup={warmup_runs} · measure={measure_runs}"
    )
    print()

    base_payload = {
        "model": "auto",
        "messages": PROMPT_QUICK,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }

    # Warmup
    for i in range(warmup_runs):
        print(f"  {DIM}warmup {i+1}/{warmup_runs}...{NC}", end="", flush=True)
        try:
            _measure_one(base_payload)
            print(f" {DIM}ok{NC}")
        except urllib.error.URLError as e:
            print(f" {YELLOW}network error: {e}{NC}")
            return 1
    print()

    ttfts: list[float] = []
    decode_rates: list[float] = []
    for i in range(measure_runs):
        try:
            ttft, total, toks = _measure_one(base_payload)
        except urllib.error.URLError as e:
            print(f"  {YELLOW}run {i+1} error: {e}{NC}")
            continue
        decode_time = max(total - ttft, 1e-6)
        rate = (toks - 1) / decode_time if toks > 1 else toks / max(total, 1e-6)
        ttfts.append(ttft)
        decode_rates.append(rate)
        print(
            f"  run {i+1}: {CYAN}{rate:>6.1f}{NC} tok/s  "
            f"{GRAY}(TTFT {ttft*1000:.0f} ms · {toks} tok in {total:.2f}s){NC}"
        )

    if not decode_rates:
        print(f"{YELLOW}All runs failed.{NC}")
        return 1

    median_rate = statistics.median(decode_rates)
    median_ttft = statistics.median(ttfts) * 1000
    print()
    print(f"{BOLD}Median:{NC}  {GREEN}{median_rate:.1f} tok/s{NC}  "
          f"{GRAY}· TTFT {median_ttft:.0f} ms{NC}")

    # Extrapolated context for users comparing to README
    if median_rate >= 50:
        verdict = f"{GREEN}healthy{NC} — your install + model are fast enough for interactive use"
    elif median_rate >= 20:
        verdict = f"{YELLOW}okay{NC} — below the 40 tok/s M5-Air-base reference (check for thermals / other load)"
    else:
        verdict = f"{YELLOW}slow{NC} — either thermal throttling, swap pressure, or an unexpectedly-slow model"
    print(f"  {verdict}")

    return 0
