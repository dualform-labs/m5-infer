# Contributing to m5-infer

Thank you for your interest in contributing! This project welcomes issues and pull requests.

## Before you start

- Open an **issue** first for significant changes (new innovations, API changes, dependency upgrades). A short discussion saves duplicated effort.
- For small bug fixes and documentation improvements, feel free to open a PR directly.

## Development setup

```bash
git clone https://github.com/dualform-labs/m5-infer.git
cd m5-infer

# Editable install with dev extras
pip install -e '.[dev]'

# Run tests
pytest tests/ -q
```

## Coding standards

- **Python 3.11+** — use modern type hints (`list[int]`, `X | None`).
- **No silent failures** — log warnings, raise explicit errors. Never `except: pass`.
- **Keep core behavior intact** — when changing `custom_generate.py` or `app/innovation/*`, run the full test suite and verify regression.
- **Model-family awareness** — if your change touches forward-pass logic, ensure it works for pure transformers (Llama / Gemma / Qwen 2.5) and hybrids (Qwen 3.5).
- **Japanese / English** — docstrings in either language are fine; user-facing log messages prefer English.

## Pull request checklist

- [ ] Tests added / updated for new behavior.
- [ ] `pytest tests/ -q` passes locally.
- [ ] No hard-coded personal paths (`/Users/...`) committed.
- [ ] No API keys, secrets, or machine identifiers in the diff.
- [ ] If touching an innovation module, the corresponding SPEC section (if public) is updated.
- [ ] Changelog entry under `[Unreleased]` in `CHANGELOG.md` for user-visible changes.

## Licensing

By submitting a Contribution, you agree that your Contribution is licensed under the **Apache License 2.0**, the same license as the project. See `LICENSE`.
