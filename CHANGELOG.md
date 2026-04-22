# Changelog

All notable changes to **m5-infer** will be documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.1.4] — 2026-04-22

Chat CLI polish for real-world daily use.

### Fixed

- **CJK / Japanese editing in `m5-infer chat`.** The built-in Python
  `input()` treated each full-width character as 1 column wide, so
  pressing Backspace after a Japanese character removed the codepoint
  but left a half-width gap where the character used to be. `chat`
  now uses `prompt_toolkit.PromptSession` for the input loop, which
  handles wide-character cursor movement correctly. Arrow-up / arrow-
  down also recalls prior prompts from a persistent history stored
  under the XDG data dir.
- **No practical input-length limit.** Some terminal TTY buffers cap
  raw-line input at ~1024 bytes; `prompt_toolkit` bypasses that.

### Added

- **`/attach <path>` command.** Stages a text file to be prepended to
  your next turn, wrapped as `<file path="...">…</file>` with a fenced
  code block. Multi-attach is supported (each `/attach` queues another
  file). Binary / non-UTF-8 files and files larger than 200 KB are
  rejected with a human-readable error. New `/attach` (no arg) lists
  the pending attachments; `/detach` clears them.

### Dependencies

- Added `prompt_toolkit>=3.0.0`. Pure Python, ~300 KB install, widely
  used (IPython, pdb++). No engine-path impact — the server never
  imports it.

### Migration

No action. `pip install --upgrade m5-infer` picks up both the dep and
the new chat behavior.

---

## [1.1.3] — 2026-04-22

Hot-fix for two P1 issues Codex review surfaced against v1.1.2 after the
PyPI push. Users on v1.1.2 should upgrade.

### Fixed

- **Chat CLI reported success on pull HTTP errors.** `m5-infer chat
  <model>` checked only `status == "error"`, but `api_post` returns
  `{"error": ..., "status_code": ...}` on HTTP failures — so a typo'd
  repo id printed "Loaded." before the chat loop died. Now treats any
  `error` key, any non-200 status_code, or an unrecognized status string
  as failure. Only `loaded`/`already_loaded` is success.
- **`POST /v1/models/pull` sync path always returned 502 on failure.**
  `/v1/models/load` correctly maps `hardware_oom` to 503 (capacity
  exceeded). Aligned the sync pull path so clients get consistent
  status codes (503 = too big for this Mac, 502 = upstream failure).
- **`_classify_pull_error` now checks HuggingFace exception classes
  before string heuristics.** `RepositoryNotFoundError` used to be
  misclassified as `auth` because HF's 401/404 conflation makes
  `str(exc)` contain "401". Correct classes (`RepositoryNotFoundError`,
  `RevisionNotFoundError`, `GatedRepoError`) are checked first, so a
  typo'd repo now surfaces as `not_found` instead of `auth`.
- **Metal OOM remediation text was inverted.** The old message said
  "raise headroom" which actually makes usable Metal memory smaller.
  Now says "lower headroom_gb or raise metal_limit_gb in engine.toml".

### Changed

- `m5-infer pull` (CLI) explicitly sends `"stream": true` in the request
  body in addition to the `Accept: text/event-stream` header, so it
  works through proxies that strip headers.
- Headroom warning messaging: < 3 GB still emits WARNING, < 4 GB now
  emits INFO (macOS needs ~4 GB for OS + active apps). Catches
  aggressive configs one step earlier.

### Docs

- `/v1/models/pull` streaming docstring clarified: response is
  `application/x-ndjson`, not SSE. Chunk schema enumerated. The
  `Accept: text/event-stream` header is still honored as an opt-in
  trigger for client convenience.

### Migration

No action required. `pip install --upgrade m5-infer`.

---

## [1.1.2] — 2026-04-22

Daily-driver CLI + first class download UX + memory-policy polish. No
engine-level changes — purely runtime / operator ergonomics.

### Added

- **`m5-infer` CLI subcommands** (7 new): `chat`, `pull`, `bench`, `stop`,
  `status`, `cache (list|clear)`, `models (--loaded|--cached)`. Replaces
  hand-rolled invocations like `python -m app.cli.chat` and the `lsof | kill`
  dance. Every subcommand speaks to the already-running server over the HTTP
  API, so `m5-infer chat` now Just Works regardless of install path.
- **Streaming `/v1/models/pull`.** Send `"stream": true` or
  `Accept: text/event-stream` to get newline-delimited JSON chunks:
  `download_start` → `downloading {bytes_dl, mbps, elapsed_s}` → `load_start`
  → `success`. Errors arrive as `{phase: "error", error_code, error}`.
  Synchronous POST still works for v1.1.0 / v1.1.1 clients.
- **Master-lock release during HF download.** The download thread in
  streaming mode runs without the inference lock, so `/health` and chat
  against the already-loaded model stay responsive while a multi-GB pull
  is underway. The lock is reacquired only for the MLX load + backend swap
  at the end.
- **`POST /v1/models/load`.** Hot-swap to a model already present in the
  HF cache without touching the network. Returns 404 with `error_code:
  not_cached` (plus a pull hint) when the repo isn't on disk. Drops
  stale in-memory CTRSP snapshots before swapping so old state can't
  collide with the new model-id.
- **Metal OOM classification.** Pull/load failures now return structured
  `error_code` (rate_limited, disk_full, auth, network, hardware_oom,
  error). `hardware_oom` specifically returns 503 ("capacity exceeded")
  so clients can distinguish "too big for this Mac" from generic 502s.
- **`GET /v1/stats`.** Lightweight metrics snapshot: MTAB hit-rate, OIRC
  stats, CTRSP on-disk size, loaded model, memory. For ad-hoc monitoring
  and `m5-infer status`. Prometheus exposition remains v1.3+ material.
- **`/health` now includes `version`, `uptime_s`, and `requests_served`**
  (additive; older clients still parse the response fine).

### Changed

- **Default Metal headroom loosened.** `auto_tune._wired_limit_from_memory`
  replaces the old flat 6/8 GB constant with a per-tier curve:

        ≤16 GB →  4 GB headroom  (was 6 → 12 GB Metal limit)
        ≤24 GB →  4 GB           (was 6 → 20 GB limit, fits 35B A3B-4bit)
        ≤32 GB →  5 GB           (was 6)
        ≤64 GB →  6 GB           (was 8)
        >64 GB →  7 GB           (was 8)

  On a 24 GB MacBook Air base the Metal limit goes 18 GB → **20 GB**,
  which lets Qwen 3.6-35B A3B-4bit (~18.6 GB) load at default settings.
- **`[memory]` config overrides.** New fields `metal_limit_gb` (absolute)
  and `headroom_gb` (subtract from total) in engine.toml, plus matching
  env vars `$M5_INFER_METAL_LIMIT_GB` / `$M5_INFER_METAL_HEADROOM_GB`.
  Startup log now tells operators exactly which knob produced the Metal
  limit:
  `Metal memory_limit set to 20480 MB (20.0 GB on 24 GB total) — source:
  auto_tune per-tier default`
- **Aggressive-limit warning.** When effective headroom drops below 3 GB
  the boot log emits a WARNING reminding operators that macOS may swap
  under memory pressure.
- **HuggingFace HTTP client log spam → WARNING.** `huggingface_hub._client`,
  `_http`, `file_download`, `httpx`, `httpcore`, and `urllib3` loggers are
  now at WARNING level in `setup_logging()`. Previous INFO-level spew was
  interleaving with every m5-infer log line during downloads.

### Migration

No breaking changes. `pip install --upgrade m5-infer` is sufficient.
Existing engine.toml continues to work; new `[memory]` fields are
optional.

---

## [1.1.1] — 2026-04-22

Emergency hot-fix for PyPI installability. **v1.1.0 (and earlier) cannot
start when installed from PyPI** because the wheel did not ship
`configs/engine.toml`, and `app.core.config` raised `FileNotFoundError`
during module import. Users running from an editable source checkout
(`pip install -e .`) were unaffected; PyPI users hit the bug immediately.

### Fixed

- **P0: PyPI install fails on first launch.** The `configs/*.toml` files
  are now embedded inside the wheel as `app/_defaults/*.toml` via hatchling
  `force-include`, and `app.core.config` no longer raises at import time
  when no project root is discoverable. See new `app/core/paths.py` for the
  full resolution order.
- **Config search order:** `$M5_INFER_CONFIG` → `./configs/*.toml` →
  `$XDG_CONFIG_HOME/m5-infer/*.toml` (falls back to `~/.config/m5-infer/`)
  → project root (editable install) → bundled `app/_defaults/*.toml`.
  Every layer logs a clear error message pointing at `m5-infer init` if
  nothing resolves.
- **Runtime data directory:** CTRSP snapshots, SQLite metrics, and logs
  used `./state/` and `./logs/` unconditionally. PyPI users running from
  `~` ended up polluting their home with these directories. Now resolved
  via `app/core/paths.py::data_root()`:
  1. `$M5_INFER_DATA_DIR` explicit override.
  2. Current working directory if it looks like the source tree
     (`pyproject.toml` + `app/` both present) — preserves the legacy layout
     for developers.
  3. `$XDG_DATA_HOME/m5-infer` (else `~/.local/share/m5-infer`).

### Added

- **`m5-infer` CLI subcommands and flags.** The console entry point now
  supports:
  - `m5-infer` / `m5-infer start` — foreground server (no `cd` needed).
  - `m5-infer init [--dir <path>]` — copy bundled defaults to
    `./configs/` for customization.
  - `m5-infer --version`, `--help`.
  - `--config <path>`, `--port <n>`, `--host <h>` runtime overrides.
- **Ready banner** on startup. A single stdout line summarizes URL, model,
  config path, data dir, and resident memory so operators see one clear
  "the server is up" marker even at default log levels.
- **First-run download notice.** When the configured main model isn't in
  the HuggingFace cache, a human-friendly warning is printed before the
  download begins so `m5-infer` no longer appears to hang silently.
- **Friendly port-in-use error.** Replaces the raw `OSError: [Errno 48]`
  with a message that suggests `--port` / `engine.toml [server] port`.
- **Fresh-venv install smoke test** added to the CI matrix so this class
  of regression is caught before future releases.

### Migration

No action required — `pip install --upgrade m5-infer` picks up the fix.
Existing editable installs continue to work unchanged. Users who had
pinned `1.1.0` and hit the bug should upgrade.

---

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
