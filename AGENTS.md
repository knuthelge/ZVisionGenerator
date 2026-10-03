# Project Guidelines

Architecture, project structure, and full conventions live in [docs/development.md](docs/development.md). Read its Project Structure and Architecture Overview sections before adding a module, backend, workflow stage, or config key. This file holds the rules every change must follow.

## Design Philosophy — Atomic Code Design

Every module, function, and class should be **atomic**: a single, self-contained unit of responsibility that can be understood, tested, and replaced in isolation.

- **One purpose per unit.** Each function does one thing. Each module owns one concept. If a docstring needs "and", split the unit.
- **Small surface area.** Expose the minimum interface necessary. Keep helpers private (`_`-prefixed). Public symbols go through `__init__.py` re-exports and `__all__`.
- **Pure over stateful.** Prefer pure functions that take inputs and return outputs. Push side effects (I/O, logging, mutation) to the edges of the system.
- **Composable building blocks.** Stages, processors, and utilities are designed to be composed into pipelines — not inherited from. Favour composition and plain function calls over class hierarchies.
- **Isolated testability.** If a unit cannot be tested without standing up half the system, it is too coupled. Every atomic unit should be testable with simple inputs and mocks.
- **Explicit dependencies.** Pass dependencies as arguments rather than importing global state. Config, backends, and I/O handles flow inward through function parameters.
- **Reuse over duplication.** Before writing new code, check `utils/` and existing helpers for something that already does the job. Extract shared behaviour into common functions rather than duplicating it across modules.

## Code Rules

- Python 3.14+ with modern type syntax (`str | None`, `list[str]`). Never use `Optional`, `Union`, `List`, `Dict`, or `Tuple` from `typing`.
- Start every `.py` file with `from __future__ import annotations`. Guard heavy imports (torch, mflux, diffusers) with `TYPE_CHECKING` or import them lazily.
- Google-style docstrings; first line is an imperative fragment.
- Backend and accelerator selection happens only in `backends/__init__.py`. Other code must not branch on platform to pick a backend.
- `@dataclass(frozen=True)` for value objects, plain `@dataclass` only for working state. No pydantic or attrs.
- Config is a plain `dict` from YAML. Precedence: CLI flags > model preset variant > model preset family > global defaults.
- Raise built-in exceptions (`ValueError`, `FileNotFoundError`, `RuntimeError`) with descriptive messages. Add a custom exception only when callers must catch it distinctly. Use `warnings.warn(..., stacklevel=2)` for non-fatal conditions.

## Repository Rules

- `packages/ltx_*` is vendored. Never edit it by hand; update it with `make update-ltx`.
- The Web UI is built from `frontend/` (Svelte 5, TypeScript, pnpm) into `zvisiongenerator/web/static/app/`, which is committed. After any frontend change, run `make frontend-build` and commit the rebuilt files.
- User-facing changes get an entry under `[Unreleased]` in `CHANGELOG.md` and updated docs in `docs/`.
- Commits follow Conventional Commits: `type(scope): imperative summary` (types: feat, fix, perf, refactor, docs, test, build, ci, chore; scope is the product area, e.g. web, image, video, enhance). Add a body with a short list of high-level changes, not file-by-file detail. Mark breaking changes with `!` and a `BREAKING CHANGE:` line. Release commits are `chore(release): v<version>`.
- Never create or push `v*` tags or cut a release unless the user explicitly asks; a tag push publishes to PyPI and cannot be undone (see [Releasing](docs/development.md#releasing)).

## Build & Test

Package manager is `uv`; run Python tools with `uv run`. Install everything with `make install`.

While iterating, run targeted checks instead of the full gate:

```bash
uv run pytest tests/test_<module>.py -k <name>   # single test file or test
make lint-fix && make format                     # auto-fix lint and formatting
make frontend-test                               # svelte-check + Vitest
```

Run `make check` before finishing. It is the full CI gate: ruff lint and format check, pytest, frontend type checks and Vitest, packaged SPA drift check, and a strict docs build.

## Testing

- Tests live in `tests/` as `test_<module>.py`; frontend tests sit next to their source as `*.test.ts`.
- Never load real backends or models. Use `MagicMock` with controlled return values and `conftest.py::_make_args(**overrides)` for CLI args.
- Test behavior and machine-readable contracts (routes, config keys, event names, enum values, statuses), not prose, help text, CSS classes, or source text.
