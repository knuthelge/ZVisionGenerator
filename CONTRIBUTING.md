# Contributing to Z-Vision Generator

Thank you for your interest in contributing!

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold it.

## Reporting Bugs

Open a [bug report](https://github.com/knuthelge/ZVisionGenerator/issues/new?template=bug_report.yml) with steps to reproduce, expected vs actual behavior, platform, Python version, and any error output.

## Suggesting Features

Open a [feature request](https://github.com/knuthelge/ZVisionGenerator/issues/new?template=feature_request.yml) describing the feature and use case.

## Submitting Code

1. Fork the repository
2. Create a feature branch (`git checkout -b my-feature`)
3. Make your changes
4. Run `make check` (lint + format-check + tests must all pass)
5. Open a Pull Request against `main`

See the [Development Guide](docs/development.md) for setup, testing, and code style details.

## Code Style

- Enforced by [ruff](https://docs.astral.sh/ruff/) (line-length 200)
- Google-style docstrings
- `from __future__ import annotations` in every `.py` file
- Python 3.14+ type syntax (`str | None`, `list[str]`)

### Frontend Development

- Node.js ≥ 22.0.0 and npm ≥ 10.0.0 are required to build and test the web frontend.
- Install frontend dependencies: `make frontend-install`
- Build frontend assets: `make frontend-build`
- Run frontend tests: `make frontend-test`
- Start Vite dev server: `make frontend-dev`

## License

By contributing, you agree that your contributions will be licensed under the [AGPL-3.0-or-later](LICENSE).
