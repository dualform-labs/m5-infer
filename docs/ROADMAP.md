# Roadmap

This document lists what we are considering for future releases. Anything **not** in this list is not being worked on. Timelines are intentionally omitted — we ship when quality is where we want it.

> Bold = high confidence. _Italic_ = exploratory / may change or be dropped.

---

## v1.0.x (patch releases, near-term)

Bug fixes and small, non-breaking improvements on top of v1.0.0.

- **Fix `sql_01`-style thinking loop** — the current n-gram detector misses loops where a ~30-token block repeats 4× (window just above the threshold). Tighten detection without triggering false positives on healthy bullet lists.
- **Extended model-family verification** — Qwen 2.5 / 3.6, Llama 3.x, Mistral, Gemma 2 / 3 / 4 are supported via the family abstraction layer but have not been benchmark-verified end-to-end. Each family will get its own smoke test.
- **16 GB memory tier support** — currently untested. Validate that small 4-bit models (3B-class) run on an entry M1 / M2 / M3 with the `moderate` memory mode.
- _Better cold-start latency_ — the first request on a fresh server is ~20 s due to RDMS draft load + Metal JIT. Explore overlapping draft load with initial prefill.

> `pip install m5-infer` is already live on PyPI (Apache 2.0). See the install section of the README.

## v1.1 — Tier validation on Pro / Max (exploratory)

The engine already ships `moderate` / `aggressive` / `extreme` memory modes, but all public bench numbers were collected on an entry M5 MacBook Air base (24 GB). Pro / Max / Studio users should see meaningfully higher numbers, but we have not verified this ourselves.

- **Bench reproduction on M5 Pro / Max / Studio** — reference data points for `extreme` mode with larger draft models (3B, 7B) and bigger CTRSP LRU.
- **48 GB+ memory profiles** — unlock larger resident models + longer prefix caches.
- _User-contributed bench results welcome_ — open an issue with your own workload metrics (decode tok/s, warm total latency, retrieval accuracy etc.) and the hardware configuration; a short OpenAI-compatible probe against `/v1/chat/completions` is sufficient.

## v1.2 — Additional decode-path optimizations (research)

Ideas from our internal roadmap that have promising PoC results but need real validation before we commit.

- _**PST — Parallel Speculative Tree**_ — extend linear lookahead into a branching decode tree (branch factor 2, depth 4). Expected: +5–8% decode when draft acceptance is high.
- _**PVS — Parallel Verification Speculative**_ — run three draft sources (RDMS draft + 2-bit shadow + n-gram) concurrently and take the longest match. Needs careful MLX stream scheduling.
- _**HGRQ — Heat-Guided Rotating Quantization**_ — dynamic KV cache precision based on attention heatmap. Expected benefit is long-context (32K+) specific; low impact on short prompts.

## v1.3 — Agent / multi-session features (exploratory)

For users running many concurrent sessions against the same engine.

- _**MRPB multi-stream activation**_ — the scheduler already splits request-class locks, but MLX multi-stream for true parallel decode is still PoC.
- _**CTRSP Delta compression**_ — sparse-delta representation of per-agent state, targeting 1024+ concurrent agents in a single resident engine.
- _**LAAS — Low-Rank Agent Adapter Swap**_ — represent system prompts as SVD low-rank deltas, enabling ~50 ms agent switching (vs ~500 ms with full CTRSP).

## v2.0 — Quality-mode + research tracks (long-term)

Large structural changes that would merit a major-version bump. No commitment to ship.

- _**FPQM — Full Precision Quality Mode**_ — per-request `kv_precision: bf16` switch so users can opt into maximal-quality output on a request-by-request basis. Useful for eval, coding, long reasoning.
- _**Vision fast path**_ — integrate Gemma 4 / other multimodal models as a resident `sub_heavy` with zero-swap routing.
- _**Dual-Path ANE + GPU inference**_ — run the 24 GDN layers on Apple Neural Engine in parallel with the 8 FA layers on GPU. Research-grade, high risk, potentially large decode gain. Will stay behind a flag if it materializes.

---

## How we prioritize

1. **Correctness and quality first.** Bug fixes and honest reporting always beat new features.
2. **Real-world value over synthetic wins.** A 2× decode gain on an artificial bench that does not survive contact with real workloads is not shipped.
3. **Measured beats theoretical.** If we cannot byte-level verify against a reference implementation, the feature does not leave research.
4. **Small, reversible releases.** Large architectural changes go through dedicated feature branches with their own benchmarks.

## Contributing to the roadmap

Have a use case that is not covered above? Open a [GitHub Discussion](https://github.com/dualform-labs/m5-infer/discussions) with:

- What you are trying to do
- Which engine you compared against and by how much m5-infer falls short
- Any relevant metrics from your own workload (decode tok/s, warm total latency, retrieval accuracy etc.); a short OpenAI-compatible probe is sufficient

We prioritize features that unblock real workloads over speculative optimizations.
