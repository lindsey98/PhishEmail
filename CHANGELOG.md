# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Community-health files: `LICENSE` (CC BY-NC 4.0), `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`, and `CITATION.cff`.
- GitHub Actions CI (ruff lint + dependency-light tests), Dependabot config,
  issue/PR templates, `CODEOWNERS`, and README badges.
- Tooling config: `pyproject.toml` (ruff/black/pytest), `.pre-commit-config.yaml`,
  `.editorconfig`.
- `examples/` with synthetic sample emails and a quickstart.
- Support for **Outlook `.msg`** files and a unified `resolve_email_input()`
  dispatcher that accepts a folder, a single `.eml`/`.txt`, or a
  `.mbox`/`.pst`/`.msg` container (directories containing archives are expanded
  and merged). Unsupported inputs now raise a clear, actionable error.
- `lib/utilities/email_io.py`: a dependency-light home for all email-format
  conversion, plus `tests/test_email_io.py`.

### Changed
- Split the 1,855-line `lib/reference_db/agent_utils.py` into focused modules:
  `agent_constants.py` (config/regex/prompts), `agent_helpers.py` (deterministic,
  network-free email/HTML/domain parsing), `agent_web.py` (HTTP fetch, BFS
  crawling, DNS), and `agent_llm.py` (OpenAI-backed phases/filters).
  `agent_utils.py` remains a thin facade that re-exports the full public surface,
  so `from .agent_utils import …` call sites are unchanged. None of
  `agent_constants`/`agent_helpers`/`agent_web` import `openai`.
- Made `lib/reference_db/__init__` import its heavy submodules
  (`IdentityMatcher`/`CharacterBert`) lazily (PEP 562), so the agent submodules
  can be imported and unit-tested without torch/faiss/transformers.

### Added
- Unit tests for the deterministic agent helpers (`tests/test_agent_helpers.py`,
  12 cases covering email extraction, Cloudflare deobfuscation, and host/domain
  normalization), runnable in CI without the ML stack.
- Cleaned up the core modules (`lib/reference_db/*`, `lib/encoder/IdentityBert.py`,
  `lib/utilities/*`, `lib/data/Dataset.py`, `lib/data/RenderDataset.py`):
  replaced the `from typing import *` star-import with explicit names, removed
  dead/duplicate imports and dead variables/comments, fixed a bare `except`,
  and applied consistent formatting — all behavior-preserving. These paths are
  now part of the enforced CI lint scope.
- Decoupled email I/O and logging from the heavy ML utilities: importing
  `lib.utilities` (and the email-conversion helpers) no longer pulls in
  `transformers`/`langdetect`/etc.
- Removed a ~257-line dead commented-out duplicate class from
  `lib/data/RenderDataset.py`.
- "How It Works" section in the README documenting the detection pipeline,
  the inference CLI options, and the `OPENAI_API_KEY` requirement for the
  knowledge-base expansion agent.
- `tests/` scaffold with structural and `label_eml` smoke tests.
- `pixi` tasks: `inference`, `serve`, `get-model`, `test`.

### Changed
- Reorganized the repository into a professional structure: entry points moved
  to `apps/cli/` and `apps/server/`; `config.py` and `label_eml.py` moved into
  the `lib/` package (`lib/config.py`, `lib/labeling/`); `get_model.sh` moved to
  `scripts/`.
- README setup now installs Chromium via Playwright (the rendering pipeline no
  longer uses `wkhtmltopdf`).
- README Output Format table corrected to match the actual CSV header.

### Fixed
- Added the missing `playwright` dependency that prevented the package from
  importing on a fresh install; made the Playwright import lazy so rendering
  degrades gracefully.
- Recipient de-duplication bug where the sender address string was iterated into
  individual characters instead of being treated as a single address.
- Knowledge-base embeddings are now reused from cache instead of being recomputed
  on every startup.
- Case-sensitivity mismatch in entity index matching during NER preprocessing.
- `IndexError` guard in inline-image reference fixing in the server.
- `label_html`/`label_headers` no longer build an empty regex alternation when
  the identities/actions list is empty (which previously matched and wrapped the
  entire document).
- Fragile email-format dispatch in the CLI (`'.mbox' in path` substring match and
  `path.replace('.mbox','')`) replaced with robust suffix handling; `.pst` now
  raises a clear "install pypff" message instead of an opaque import error.

[Unreleased]: https://github.com/your-org/PhishEmail/commits/main
