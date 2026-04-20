---
name: Bug report
about: Report unexpected behavior or a crash
title: "[BUG] "
labels: bug
assignees: ''
---

## Summary
<!-- One sentence: what did you expect vs what happened? -->

## Reproduction
<!-- Minimum steps to reproduce. If possible, include a curl command or minimal Python snippet. -->

1.
2.
3.

## Expected behavior
<!-- What should have happened. -->

## Actual behavior
<!-- What actually happened. Include exact error messages / stack traces. -->

## Environment

- **Chip**: <!-- e.g. Apple M2 Pro 10-core GPU -->
- **Unified memory**: <!-- e.g. 32 GB -->
- **macOS**: <!-- e.g. 26.4 -->
- **Python**: <!-- e.g. 3.12.8 -->
- **m5-infer version**: <!-- e.g. v1.0.0 / commit SHA -->
- **Model**: <!-- e.g. mlx-community/Qwen3.5-9B-MLX-4bit -->
- **Runtime mode** (`[runtime] memory_mode` in engine.toml): <!-- moderate / aggressive / extreme -->

## Engine logs
<details>
<summary>`logs/m5-infer.log` tail (last ~100 lines)</summary>

```
<paste here>
```

</details>

## Additional context
<!-- Anything that might help: custom config, concurrent request load, specific prompt patterns, etc. -->
