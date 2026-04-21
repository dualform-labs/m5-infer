---
name: Benchmark report
about: Share bench numbers from your hardware (Pro / Max / Studio welcome)
title: "[BENCH] "
labels: benchmark
assignees: ''
---

## Hardware

- **Mac**: <!-- e.g. Mac Studio M2 Ultra, MacBook Pro M4 Max 16-core GPU, ... -->
- **Unified memory**: <!-- e.g. 128 GB -->
- **Cooling**: <!-- fanless (Air) / fan / external -->
- **macOS**: <!-- e.g. 26.4 -->

## Model

- **Main path**: <!-- e.g. mlx-community/Qwen3.5-9B-MLX-4bit -->
- **Draft path** (if any): <!-- e.g. mlx-community/Qwen3.5-0.8B-MLX-4bit -->
- **Runtime mode**: <!-- moderate / aggressive / extreme -->

## Numbers

Run one or more standard scripts and paste the output (whichever are relevant):

```
# Any bench suite you have. Values we care about:
#   - decode tok/s (long_gen 512 tokens)
#   - Warm total latency / TTFT (2nd call with same system prompt; clarify which you measured)
#   - Session turn-5 latency (5-turn conversation)
#   - Short QA pass rate (thinking ON / OFF)
#   - Needle retrieval at 4K / 12K / 32K context
```

## Comparison

If you compared against other engines on the same Mac (mlx_lm.server, Ollama, llama.cpp, etc.), include their numbers for apples-to-apples:

| Engine | Decode tok/s | Warm total latency | Notes |
|:---|---:|---:|:---|
|     |     |     |     |

## Observations
<!-- Anything unusual: thermal behavior, memory pressure events, etc. -->
