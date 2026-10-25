# Contributing to Z-Vision Generator

Thank you for your interest in contributing!

The full guide is in the [developer docs](https://knuthelge.github.io/ZVisionGenerator/developer/) (or [docs/developer/](docs/developer/index.md) in this repository): setting up, the conventions every change follows, and how the app is built.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold it.

## Reporting Bugs

Open a [bug report](https://github.com/knuthelge/ZVisionGenerator/issues/new?template=bug_report.yml) with steps to reproduce, expected vs actual behavior, platform, Python version, and any error output. Report security issues as described in [SECURITY.md](SECURITY.md).

## Suggesting Features

Open a [feature request](https://github.com/knuthelge/ZVisionGenerator/issues/new?template=feature_request.yml) describing the feature and use case.

## Submitting Code

1. Fork the repository and create a feature branch.
2. Set up with `make install`. See [Set up](docs/developer/index.md#set-up).
3. Make your changes, following the [conventions](docs/developer/conventions.md), with tests.
4. Update the docs and `CHANGELOG.md` for user-facing changes. See [Writing docs](docs/developer/docs.md).
5. Run `make check`: lint, format check, pytest, frontend checks, the packaged SPA drift check and a strict docs build must all pass.
6. Commit with [Conventional Commits](https://www.conventionalcommits.org/) messages (`type(scope): imperative summary`, plus a short list of high-level changes in the body) and open a pull request against `main`.

## License

By contributing, you agree that your contributions will be licensed under the [AGPL-3.0-or-later](LICENSE).
