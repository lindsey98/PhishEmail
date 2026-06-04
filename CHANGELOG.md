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

[Unreleased]: https://github.com/your-org/PhishEmail/commits/main
