# Conventions

The code rules every change follows. `AGENTS.md` in the repository holds the same rules for coding agents; repository, commit and release rules are on the [Developer](index.md#repository-rules) page.

## Design: atomic units

Every module, function and class should be a single, self-contained unit of responsibility that can be understood, tested and replaced on its own.

- **One purpose per unit.** Each function does one thing; each module owns one concept. If a docstring needs "and", split the unit.
- **Small surface.** Expose the minimum. Keep helpers private (`_`-prefixed); public symbols go through `__init__.py` re-exports and `__all__`.
- **Pure over stateful.** Prefer functions that take inputs and return outputs. Push side effects (I/O, logging, mutation) to the edges.
- **Composition over inheritance.** Stages, processors and utilities are composed into pipelines, not inherited from.
- **Testable in isolation.** If a unit can't be tested without standing up half the system, it is too coupled.
- **Explicit dependencies.** Pass config, backends and I/O handles in as arguments rather than importing global state.
- **Reuse over duplication.** Check `utils/` and existing helpers before writing new code.

## Python

- Python 3.14+. Start every `.py` file with `from __future__ import annotations`.
- Modern type syntax: `str | None`, `list[str]`, `dict[str, Any]`. Never `Optional`, `Union`, `List`, `Dict` or `Tuple` from `typing`.
- Imports: standard library, third-party, then `zvisiongenerator.*`, with a blank line between groups.
- Guard heavy imports (torch, mflux, diffusers) with `TYPE_CHECKING`, or import them lazily.
- Line length 200, enforced by ruff.
- Google-style docstrings. Module docstrings on every non-trivial module. The first line is an imperative fragment; longer ones use `Args:`, `Returns:` and `Raises:`.

## Platform and backends

Backend and accelerator selection happens only in `backends/__init__.py`; other code asks a capability function such as `supports_video_extend()`. What else is fine, and the two platform key schemes (`darwin`/`win32`/`linux` and `cuda`/`mps`), are in the [platform rule](architecture.md#invariants). See [Backends](backends.md).

## Data types

`@dataclass(frozen=True)` for value objects (inputs, detection results). Plain `@dataclass` only for working state. No pydantic or attrs.

## Config

Config is a plain `dict` loaded from YAML, not a dataclass. Precedence: CLI flags, then the model preset's variant, then the model preset's family, then global defaults. See [Configuration](../reference/configuration.md).

## Errors

Raise `ValueError`, `FileNotFoundError` or `RuntimeError` with a descriptive message. Use `warnings.warn(..., stacklevel=2)` for non-fatal conditions. Add a custom exception only when callers must catch it distinctly, such as `JobConflictError` in `web/web_runner.py`, or as a private sentinel.

Treat the wording of an error a user can see as part of the interface. `tests/common/test_docs_troubleshooting.py` fails when a quoted heading of the [Troubleshooting](../reference/troubleshooting.md) page no longer matches a message in the code, so rewording an error means updating that heading in the same change.

## Tests

Every change comes with tests, in `tests/<area>/test_<module>.py`. Never load a real backend or model: use `MagicMock` and `tests/_factories.py::_make_args(**overrides)`. Test behavior and machine-readable contracts (routes, config keys, event names, enum values, statuses), not prose, help text, CSS classes or source text. See [Testing](testing.md) for the areas, the guards every test runs under and which check to run when.

## Web UI

The frontend has its own conventions for confirmations, toasts, requests and its design system. See [Frontend](frontend.md).
