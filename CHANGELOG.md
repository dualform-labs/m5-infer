# Changelog

All notable changes to **m5-infer** will be documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.1.0] — 2026-04-22

Quality-neutral critical-path optimization stack plus agent-safe opt-in
response caching. All changes are byte-equivalent to v1.0.0 greedy
decoding (`temperature == 0`); no output quality trade-offs were accepted.

### Added — critical-path optimization stack (T2–T12)

All active by default, no user-visible API changes:

- **T2 CPP — Critical Path Pruning.** Per-request INFO-level logs moved to
  DEBUG. Saves ~10–30 ms per request on the hot path.
- **T6 PLSP — Prefill Log Suppression.** Chunk-progress logs moved to
  DEBUG. Saves 20–80 ms at ISL 12K prefill.
- **T7 CPSB — CTRSP Short-Prompt Bypass.** Skips CTRSP snapshot/save when
  `sys_tokens < 1024` (never worth the I/O for short prompts). Saves up
  to ~1400 ms on short-prompt requests that previously paid the CTRSP
  cost unnecessarily.
- **T8 Async CTRSP Save.** Moves the 52 MB `np.savez` persist call to a
  background worker thread; dropped gzip compression which was CPU-bound
  on the critical path. Removes 500–2000 ms of blocking at end-of-request.
- **T9 Native-Dtype State Cache.** Keeps recurrent state in bf16 in RAM;
  fp32 conversion only happens at the disk boundary. Saves ~500 ms per
  cache-hit.
- **T10 Respect Explicit Thinking Override.** Heuristic that auto-enables
  thinking mode on long-system + short-user prompts now honors explicit
  `enable_thinking=false` from the caller. Important for reproducible
  benchmarks.
- **T11 Zero-Copy CTRSP Injection.** Skip `astype` when the cached state
  dtype already matches the model's expected dtype. Saves 32–80 ms at
  12K cache-hit.
- **T12 FPTC — Full-Prompt Token Cache.** LRU (32 entries) keyed on
  `(messages_hash, tools_hash, enable_thinking)` so we don't retokenize
  the full prompt when the same request repeats. Saves 30–100 ms per
  repeat.

### Added — T14-OIRC (Opt-In Response Cache, agent-safe)

New `app/engine/oirc.py` module and two optional fields on
`ChatCompletionRequest`: `idempotency_key: str | None` and
`cache_ttl_ms: int | None`. Full-response replay (~1 ms) only activates
when **both** fields are supplied **and** the request is deterministic
(`temperature == 0`, `top_p >= 1`, no `tools`). Default behavior is
cache-off, which preserves AI-agent "re-check current state" workflows
(sending the same probe repeatedly to observe fresh state). Semantics
follow the Stripe / Plaid idempotency-key pattern.

### Changed

- `FastAPI` OpenAPI `version` metadata now reads from `app.__version__`
  instead of a hardcoded string. Keeps `/v1/models` and `/openapi.json`
  output in sync with `pyproject.toml` on every release.
- Public git excludes `app/bench/`, `docs/figures/`, `docs/v1.1_*`,
  `docs/OPUS_QUALITY_REPORT_*`, and bench raw JSON files. Operational
  guides (`MODEL_FAMILY_GUIDE.md`, `ROADMAP.md`) remain public.

### Docs

- Bench labeling cleanup: the previously-named "Warm TTFT" metric is
  measured as end-to-end completion time for a 30-token response, not
  strict TTFT per the vLLM/SGLang definition. Renamed to "Warm total
  latency" throughout README, site, figures, and the issue template.
  Strict vLLM-compatible TTFT re-measurement is planned for v1.2.
- Speedup claims now carry explicit "up to" / "最大" qualifiers, with
  workload-bound parenthetical (long_gen 512-token decode) since these
  are peak values, not universal multipliers.

### Removed

- **T14 DRC (Deterministic Response Cache)** — shipped internally but
  pulled before public release. Caching deterministic request/response
  pairs broke AI-agent "re-check current state" workflows: the agent
  sends the same probe to observe fresh state, DRC returned byte-identical
  cached output instead. Replaced by the agent-safe opt-in T14-OIRC
  described above.

### Compatibility

- Python 3.11, 3.12, 3.13, and 3.14. MLX ships cp311–cp314 wheels on PyPI.
- Apple Silicon (M1 / M2 / M3 / M4 / M5).
- `mlx >= 0.31.1`, `mlx-lm >= 0.21.0`.

### Migration from v1.0.0

No breaking changes. Upgrade with `pip install --upgrade m5-infer`. All
existing `configs/engine.toml` settings and OpenAI-compatible client
code continue to work unchanged.

---

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
- Python 3.11, 3.12, 3.13, and 3.14.
- Apple Silicon (M1 / M2 / M3 / M4 / M5).
- `mlx >= 0.31.1`, `mlx-lm >= 0.21.0`.

### Notes
- Qwen 3.5 hybrid is the primary reference architecture; other families are scaffolded but require verification against user workloads.
- Benchmarks are intentionally excluded from the public tree; users are encouraged to reproduce on their own hardware.
