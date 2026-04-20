# Changelog

All notable changes to **m5-infer** will be documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-04-19

First public release.

### Added
- **Apache License 2.0** — initial open-source license.
- **Hardware auto-tune** (`app/core/hardware_profile.py`, `app/core/auto_tune.py`). Detects Apple Silicon chip family/variant, GPU/ANE cores, unified memory, and bandwidth at startup; applies per-tier defaults (entry/standard/high/ultra) so M1 base through M5 Ultra all run near-optimal without manual tuning. Toggle via `[engine] auto_tune`.
- **Minimum requirements enforcement** — M1 or newer Apple Silicon, macOS 26.4+, 16 GB unified memory (24 GB+ recommended for 9B-class 4-bit models). Warnings are logged for below-recommended configurations; startup still proceeds for compatibility.
- **Model-family abstraction layer** (`app/core/model_family.py`, `app/backend/mask_adapter.py`). Auto-detection for Qwen 3.5 / 3.6 / 2.5, Llama 3.x, Mistral, Gemma 2/3/4. Pure-transformer families enable RDMS speculative fast path.
- `[model]` section in `configs/engine.toml` for family / main_path / draft_path selection.
- `docs/MODEL_FAMILY_GUIDE.md` — guide for adding new model families without touching core code.
- **Needle-retrieval heuristic** in `app/api/routes.py` — long-system + short-user query detection automatically enables thinking mode, avoiding Qwen safety-alignment refusals on long-context retrieval.

### Core innovations (carried over from previous internal iterations)
- **N1 CTRSP** — cross-turn recurrent state persistence for GatedDeltaNet.
- **TPC** — Token Prefix Compiler based on raw-bytes SHA-256 fingerprints.
- **Think-Aware Budget** — separate budgets for reasoning and answer tokens.
- **Loop-Escape Injection** — safe `</think>` injection on detected reasoning loops.
- **N3 SSEE** — self-speculative early exit.
- **N4 ALS** — adaptive layer skipping.
- **N5 ERP** — entropy-routed pruning.
- **N6 PES** — parallel expert scheduling.
- **X2 DPC** — dynamic precision cascading.
- **X4 Context Fold** — prefix-aware context compression.
- **X5-R** — compiled forward via `mx.compile`.
- **RDMS** — resident draft-model speculative decoding (hybrid + pure transformer paths, double-verify safety).

### Compatibility
- Python 3.11 and 3.12.
- Apple Silicon (M1 / M2 / M3 / M4 / M5).
- `mlx >= 0.31.1`, `mlx-lm >= 0.21.0`.

### Notes
- Qwen 3.5 hybrid is the primary reference architecture; other families are scaffolded but require verification against user workloads.
- Benchmarks are intentionally excluded from the public tree; users are encouraged to reproduce on their own hardware.
