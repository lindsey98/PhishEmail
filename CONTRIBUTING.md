# Contributing to PiMRef

Thanks for your interest in improving PiMRef! This document explains how to set
up your environment and submit changes. By participating you agree to abide by
our [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- **Report bugs** — open an issue with steps to reproduce, the commit hash, and
  your environment (OS, GPU/CPU, `pixi` version).
- **Suggest features or improvements** — open an issue describing the use case.
- **Submit fixes / features** — open a pull request (see below).
- **Improve documentation** — typo fixes and clarifications are very welcome.

For anything large or architectural, please open an issue to discuss the design
before investing significant effort.

## Development setup

This project uses [Pixi](https://pixi.sh) for environment management. All commands
are run **from the project root**.

```bash
# 1. Install dependencies
pixi install
pixi run playwright install chromium

# 2. (Optional) download model checkpoints
pixi run get-model     # or: bash scripts/get_model.sh

# 3. Run the test suite
pixi run test
```

See the [README](README.md) for the full setup, project structure, and how the
detection pipeline works.

## Project layout

- `apps/` — runnable entry points (`cli/inference.py`, `server/app.py`).
- `lib/` — the importable package (parsing, models, knowledge base, baselines).
- `scripts/` — operational scripts.
- `tests/` — the test suite.

When adding code, keep modules inside `lib/` and expose entry points via `apps/`.
Resource paths are resolved relative to the project root.

## Pull request guidelines

1. **Fork & branch.** Create a topic branch from `main`
   (e.g. `fix/eml-parsing-crash` or `feat/new-baseline`).
2. **Keep PRs focused.** One logical change per PR is much easier to review.
3. **Add/Update tests** where practical. Prefer dependency-light tests so they
   run without the heavy GPU/ML stack (see `tests/` for examples using
   `pytest.importorskip`).
4. **Run the tests** locally: `pixi run test`.
5. **Update documentation** (README, docstrings) for any user-facing change.
6. **Write clear commit messages** — a concise summary line plus a body
   explaining the *why*.
7. **Describe your PR** — what changed, why, and how you verified it. Link any
   related issue.

## Coding conventions

- Target Python ≥ 3.8 (matching `pixi.toml`).
- Match the style of the surrounding code; prefer type hints and docstrings on
  public functions.
- Avoid hard-coding absolute paths; keep resource paths relative to the project
  root (or `__file__` where appropriate).
- Don't commit large artifacts (checkpoints, datasets, rendered images). These
  are covered by `.gitignore`.

## Reporting security issues

Please do **not** file public issues for security vulnerabilities. Follow the
process in [SECURITY.md](SECURITY.md) instead.

## License

By contributing, you agree that your contributions will be licensed under the
project's [CC BY-NC 4.0](LICENSE) license.
