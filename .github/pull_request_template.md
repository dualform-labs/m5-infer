<!-- Thanks for contributing! Before submitting, please skim this template. Anything not applicable: delete it. -->

## Summary
<!-- What does this PR do, in one sentence? -->

## Why
<!-- What problem does it solve? Link to an issue if one exists. -->

## Changes
<!-- High-level list of changes. No need to repeat the diff verbatim. -->

-
-
-

## Test plan

- [ ] `pytest tests/ -q` passes locally
- [ ] CI is green
- [ ] If this touches `app/backend/custom_generate.py` or an innovation module, I ran at least one smoke bench (`./chat.sh` or `curl /v1/chat/completions`) and verified output matches a known-good baseline
- [ ] If this changes decode behavior, I verified byte-level output equivalence against `mlx_lm.generate` with `temperature=0` on at least one prompt

## Checklist

- [ ] No hard-coded personal paths (`/Users/...`), API keys, or machine identifiers in the diff
- [ ] No internal version labels (`v3.0`, `v0.1`, `Phase X`) or unreleased-tech names in user-facing text
- [ ] `CHANGELOG.md` updated under `[Unreleased]` for user-visible changes
- [ ] README / docs updated if the change affects configuration, the OpenAI API surface, or supported hardware

## Compatibility

- **Model families**: <!-- Qwen 3.5 only / all families / not applicable -->
- **Memory modes**: <!-- moderate / aggressive / extreme / all -->
- **Breaking change?**: <!-- no / yes — describe -->

## Additional notes
<!-- Anything a reviewer should know that isn't obvious from the diff. -->
