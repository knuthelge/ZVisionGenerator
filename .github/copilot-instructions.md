# Project Guidelines

## Design Philosophy — Atomic Code Design

Every module, function, and class should be **atomic**: a single, self-contained unit of responsibility that can be understood, tested, and replaced in isolation.

- **One purpose per unit.** Each function does one thing. Each module owns one concept. If a docstring needs "and", split the unit.
- **Small surface area.** Expose the minimum interface necessary. Keep helpers private (`_`-prefixed). Public symbols go through `__init__.py` re-exports.
- **Pure over stateful.** Prefer pure functions that take inputs and return outputs. Push side effects (I/O, logging, mutation) to the edges of the system.
- **Composable building blocks.** Stages, processors, and utilities are designed to be composed into pipelines — not inherited from. Favour composition and plain function calls over class hierarchies.
- **Isolated testability.** If a unit cannot be tested without standing up half the system, it is too coupled. Every atomic unit should be testable with simple inputs and mocks.
- **Explicit dependencies.** Pass dependencies as arguments rather than importing global state. Config, backends, and I/O handles flow inward through function parameters.
- **Reuse over duplication.** If logic already exists as a helper or utility, use it. Extract shared behaviour into common functions rather than duplicating code across modules. Before writing new code, check `utils/` and existing helpers for something that already does the job.

## Code Style

- Python 3.14+. Use modern type syntax: `str | None`, `list[str]`, `dict[str, Any]`. Never use `Optional`, `Union`, `List`, `Dict`, or `Tuple` from `typing`.
- Start every `.py` file with `from __future__ import annotations`.
- Import order: stdlib → third-party → local (`zvisiongenerator.*`). Separate each group with a blank line.
- Use `TYPE_CHECKING` guards for imports that pull in heavy runtime dependencies (torch, mflux, diffusers).
- Line length limit: 200 (enforced by ruff).
- Docstrings: Google-style. Module-level docstrings on all non-trivial modules. First line is an imperative fragment. Multi-line use `Args:`, `Returns:`, `Raises:` sections.

## Architecture

- **Backend Protocol**: Platform backends live in `backends/image_mac.py` (mflux/MLX) and `backends/image_win.py` (diffusers/CUDA). Image backends implement the `ImageBackend` Protocol from `core/image_backend.py`; video backends implement the `VideoBackend` Protocol from `core/video_backend.py`. Platform selection uses `sys.platform` in `backends/__init__.py` — never check platform elsewhere.
- **Workflow stages**: Image stage functions in `workflows/image_stages.py` have the uniform signature `(ImageGenerationRequest, ImageWorkingArtifacts) -> StageOutcome`. Video stage functions in `workflows/video_stages.py` have the signature `(VideoGenerationRequest, VideoWorkingArtifacts) -> StageOutcome`. Stages are composed dynamically by `build_workflow()` and `build_video_workflow()`.
- **Data types**: Use `@dataclass(frozen=True)` for immutable value objects (inputs, detection results). Use mutable `@dataclass` only for working state. No pydantic or attrs.
- **Config layering**: CLI flags > model preset variant > model preset family > global defaults. Config is a plain `dict` loaded from YAML, not a dataclass.
- **Errors**: Raise `ValueError`, `FileNotFoundError`, `RuntimeError` directly with descriptive f-string messages. Use `warnings.warn()` with `stacklevel=2` for non-fatal conditions. No custom exception classes except private sentinels.
- **`__init__.py` re-exports**: Subpackage `__init__.py` files re-export public symbols via `__all__` (see `utils/__init__.py`). When adding a new public function or class to a subpackage, add it to the corresponding `__init__.py` and `__all__`.
- **Package name**: The pip package is `z-vision-generator` but the Python package directory is `zvisiongenerator/`. All internal imports use `zvisiongenerator.` prefix.

## Build & Test

```bash
make install    # Install dependencies (uv sync)
make lint       # Ruff linter
make format     # Ruff formatter
make test       # pytest
make check      # Full CI gate: lint + format-check + test
```

Package manager is `uv`. Run tools with `uv run`.

After making changes, run `make check` to validate (lint + format-check + test).

## Testing Conventions

- Tests live in `tests/`. Name files `test_<module>.py`, functions `test_<behavior>`. Group related tests in `Test`-prefixed classes.
- Never import or instantiate real backends/models. Use `unittest.mock.MagicMock` with controlled return values. Helper pattern: `_make_mock_backend()` returning a MagicMock with `.text_to_image.return_value = Image.new("RGB", (64, 64))`.
- Use `conftest.py::_make_args(**overrides)` to build minimal `argparse.Namespace` objects.
- Skip heavy dependencies: `pytest.importorskip("torch")`, `@pytest.mark.skipif(sys.platform == ...)` for platform-specific tests.
- Use `tmp_path` fixture for filesystem operations.
- Use `@pytest.mark.parametrize` for data-driven tests with many input/output pairs.
